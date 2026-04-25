mod action;
mod confirmation;
mod memory;
mod prompt;
mod schema;
mod tool_local;
mod tool_remote;
mod tool_runner;
pub mod tools;
mod transcript;
mod workspace;

use crate::{
    config::Config,
    model_client::{ModelClient, ModelMessage},
};
use uuid::Uuid;

use action::Action;
use memory::MemoryStore;
pub use schema::{event, filter_events, response, ChatRequest, ChatResponse, Event};
pub use transcript::TranscriptStore;

#[derive(Clone)]
pub struct Agent {
    config: Config,
    store: TranscriptStore,
    memory: MemoryStore,
    model: ModelClient,
}

impl Agent {
    pub fn new(config: Config, model: ModelClient) -> Self {
        let _ = std::fs::create_dir_all(&config.tool_workspace_dir);
        Self {
            store: TranscriptStore::new(config.runs_dir()),
            memory: MemoryStore::new(config.memory_path()),
            config,
            model,
        }
    }
    pub async fn chat(&self, request: ChatRequest) -> ChatResponse {
        let run_id = request.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let max_steps = request.max_steps.unwrap_or(self.config.agent_max_steps);
        let visible = request.visible_event_kinds.clone();
        let mut events = vec![event("user", request.message.clone(), None, None)];
        if !self.model.is_reachable().await {
            events.push(event("error", "model server unreachable".into(), None, None));
            self.persist(&run_id, &events);
            return response(run_id, "model unavailable".into(), filter_events(&events, &visible), "model_error");
        }
        let base_prior = self.transcript(&run_id).unwrap_or_default();
        let mut prior = base_prior.clone();
        prior.extend(events.clone());
        let mut assistant = String::new();
        let mut stop_reason = "max_steps".to_string();
        let mut last_action = String::new();
        for step in 1..=max_steps {
            let action = match self.next_action(&run_id, &prior, step).await {
                Ok(action) => action,
                Err(error) => {
                    events.push(event("error", error.clone(), None, Some(step)));
                    stop_reason = if error.starts_with("model request failed")
                        || error.starts_with("model server returned")
                        || error.starts_with("model response parse failed")
                    {
                        "model_error".into()
                    } else {
                        "invalid_action".into()
                    };
                    break;
                }
            };
            let signature = action.signature();
            if action.tool != "agent.finish" && signature == last_action {
                events.push(event("error", "repeated identical non-terminal action".into(), None, Some(step)));
                stop_reason = "repeat_action".into();
                break;
            }
            last_action = signature;
            if let Some(reasoning) = action.reasoning.clone().filter(|v| !v.is_empty()) {
                events.push(event("reasoning", reasoning, None, Some(step)));
            }
            match action.tool.as_str() {
                "agent.finish" => {
                    match tools::ToolCall::from_fields(&action) {
                        Ok(tools::ToolCall::AgentFinish { content }) => {
                            events.push(event("finish", content.clone(), Some("agent.finish".into()), Some(step)));
                            assistant = content.clone();
                            events.push(event("assistant", content, None, Some(step)));
                            stop_reason = "finish".into();
                            break;
                        }
                        Err(error) => {
                            events.push(event("error", error, None, Some(step)));
                            stop_reason = "invalid_action".into();
                            break;
                        }
                        _ => unreachable!(),
                    }
                }
                "agent.think" => {
                    match tools::ToolCall::from_fields(&action) {
                        Ok(tools::ToolCall::AgentThink { content }) => {
                            events.push(event("plan", content, Some("agent.think".into()), Some(step)));
                            prior = base_prior.clone();
                            prior.extend(events.clone());
                            if step == max_steps {
                                stop_reason = "max_steps".into();
                            }
                        }
                        Err(error) => {
                            events.push(event("error", error, None, Some(step)));
                            stop_reason = "invalid_action".into();
                            break;
                        }
                        _ => unreachable!(),
                    }
                }
                "agent.request_confirmation" => {
                    match confirmation::handle(action, step, &mut events) {
                        Ok(message) => {
                            assistant = message;
                            stop_reason = "confirmation_required".into();
                            break;
                        }
                        Err(error) => {
                            events.push(event("error", error, None, Some(step)));
                            stop_reason = "invalid_action".into();
                            break;
                        }
                    }
                }
                _ => {
                    let result = self.action_tool(action, &run_id, step, &mut events).await;
                    prior = base_prior.clone();
                    prior.extend(events.clone());
                    if let Err(error) = result {
                        events.push(event("error", error, None, Some(step)));
                        stop_reason = "invalid_action".into();
                        break;
                    }
                    if step == max_steps {
                        stop_reason = "max_steps".into();
                    }
                }
            }
        }
        if assistant.is_empty() {
            assistant = format!("agent stopped: {stop_reason}");
        }
        self.persist(&run_id, &events);
        response(run_id, assistant, filter_events(&events, &visible), &stop_reason)
    }
    pub fn transcript(&self, run_id: &str) -> Result<Vec<Event>, std::io::Error> { self.store.read(run_id) }
    async fn next_action(
        &self,
        run_id: &str,
        events: &[Event],
        step: usize,
    ) -> Result<Action, String> {
        let messages = self.prompt(run_id, events, step);
        let text = self.model.chat(&messages).await?;
        match action::parse(&text) {
            Ok(action) => Ok(action),
            Err(error) if self.config.agent_repair_attempts > 0 => {
                let mut repair = messages;
                repair.push(ModelMessage {
                    role: "user".into(),
                    content: format!("Repair invalid XML action. Error: {error}. Text: {text}"),
                });
                action::parse(&self.model.chat(&repair).await?)
            }
            Err(error) => Err(error),
        }
    }

    async fn action_tool(
        &self,
        action: Action,
        run_id: &str,
        step: usize,
        events: &mut Vec<Event>,
    ) -> Result<(), String> {
        let tool = action.tool.clone();
        let call = tools::ToolCall::from_fields(&action)?;
        let result = tool_runner::run(call, &self.config, &self.memory, run_id, step, events).await;
        events.push(event("observation", result, Some(tool), Some(step)));
        Ok(())
    }

    fn prompt(&self, run_id: &str, events: &[Event], step: usize) -> Vec<ModelMessage> {
        prompt::build(run_id, events, step, &self.memory)
    }

    fn persist(&self, run_id: &str, events: &[Event]) {
        let _ = self.store.append_many(run_id, events);
    }
}
