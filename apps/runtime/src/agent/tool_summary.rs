use super::tools::ToolCall;

pub fn summary(call: &ToolCall) -> String {
    match call {
        ToolCall::AgentFinish { content } => content.chars().take(80).collect(),
        ToolCall::AgentThink { content } => content.chars().take(80).collect(),
        ToolCall::Shell { command } => command.clone(),
        ToolCall::Fetch { url } => url.clone(),
        ToolCall::Read { path } | ToolCall::List { path } => path.clone(),
        ToolCall::Write { path, content } => format!("{path} ({} bytes)", content.len()),
        ToolCall::MemorySearch { query } => query.clone(),
        ToolCall::MemoryWrite { content } => content.clone(),
        ToolCall::ResourceSearch { query, kind, .. } => format!("{query} [{kind}]"),
        ToolCall::ResourceFetch { reference } | ToolCall::ResourceHistory { reference } => {
            reference.clone()
        }
        ToolCall::ResourcePreview { body, .. } | ToolCall::ResourceCreateNote { body, .. } => {
            body.chars().take(80).collect()
        }
        ToolCall::ResourceCreateMedia { path, .. } => path.clone(),
        ToolCall::ResourceUpdate {
            reference, body, ..
        } => format!("{reference}: {}", body.chars().take(60).collect::<String>()),
    }
}
