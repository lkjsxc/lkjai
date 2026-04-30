mod action;
mod chat;
mod chat_actions;
mod confirmation;
mod confirmation_flow;
mod memory;
mod prompt;
mod schema;
mod tool_fields;
mod tool_local;
mod tool_remote;
mod tool_runner;
mod tool_summary;
pub mod tools;
mod transcript;
mod workspace;

use crate::{
    config::Config,
    model_client::{ModelClient, ModelMessage},
};

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
    pub fn transcript(&self, run_id: &str) -> Result<Vec<Event>, std::io::Error> {
        self.store.read(run_id)
    }
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
