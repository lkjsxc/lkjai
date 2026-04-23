use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub struct PendingToolCall {
    pub tool: String,
    pub args: Value,
}

#[derive(Debug, Deserialize)]
pub struct Action {
    pub kind: String,
    pub thought: Option<String>,
    pub tool: Option<String>,
    pub args: Option<Value>,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub operation: Option<String>,
    pub pending_tool_call: Option<PendingToolCall>,
}

pub fn parse(text: &str) -> Result<Action, String> {
    serde_json::from_str(text.trim()).map_err(|error| format!("invalid action json: {error}"))
}
