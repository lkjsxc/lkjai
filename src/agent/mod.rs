mod tools;
mod transcript;

use crate::{config::Config, inference::Generator};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use transcript::TranscriptStore;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub run_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub run_id: String,
    pub assistant: String,
    pub events: Vec<Event>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub kind: String,
    pub content: String,
    pub tool: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone)]
pub struct Agent {
    config: Config,
    store: TranscriptStore,
    generator: Generator,
}

impl Agent {
    pub fn new(config: Config, generator: Generator) -> Self {
        Self {
            store: TranscriptStore::new(config.runs_dir()),
            config,
            generator,
        }
    }

    pub async fn chat(&self, request: ChatRequest) -> ChatResponse {
        let run_id = request.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let mut events = vec![event("user", request.message.clone(), None)];
        let assistant = match tools::parse_tool(&request.message) {
            Some(call) => self.run_tool(call, &mut events).await,
            None => self.generator.generate(&request.message).await,
        };
        events.push(event("assistant", assistant.clone(), None));
        let _ = self.store.append_many(&run_id, &events);
        ChatResponse {
            run_id,
            assistant,
            events,
        }
    }

    pub fn transcript(&self, run_id: &str) -> Result<Vec<Event>, std::io::Error> {
        self.store.read(run_id)
    }

    async fn run_tool(&self, call: tools::ToolCall, events: &mut Vec<Event>) -> String {
        events.push(event("tool_call", call.summary(), Some(call.name().into())));
        let result = tools::execute(call, &self.config).await;
        let content = match result {
            Ok(output) => output,
            Err(error) => format!("tool failed: {error}"),
        };
        events.push(event("tool_result", content.clone(), None));
        content
    }
}

pub fn event(kind: &str, content: String, tool: Option<String>) -> Event {
    Event {
        kind: kind.into(),
        content,
        tool,
        timestamp: Utc::now(),
    }
}
