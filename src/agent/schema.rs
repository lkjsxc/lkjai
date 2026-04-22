use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub run_id: Option<String>,
    pub max_steps: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub run_id: String,
    pub assistant: String,
    pub events: Vec<Event>,
    pub stop_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub kind: String,
    pub content: String,
    pub tool: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub step: Option<usize>,
}

pub fn event(kind: &str, content: String, tool: Option<String>, step: Option<usize>) -> Event {
    Event {
        kind: kind.into(),
        content,
        tool,
        timestamp: Utc::now(),
        step,
    }
}

pub fn response(run_id: String, assistant: String, events: Vec<Event>, stop: &str) -> ChatResponse {
    ChatResponse {
        run_id,
        assistant,
        events,
        stop_reason: stop.into(),
    }
}
