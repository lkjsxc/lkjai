use crate::config::Config;

use super::{event, memory::MemoryStore, tools, Event};

pub async fn run(
    call: tools::ToolCall,
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
    step: usize,
    events: &mut Vec<Event>,
) -> String {
    let tool = call.name().to_string();
    events.push(event(
        "tool_call",
        call.summary(),
        Some(tool.clone()),
        Some(step),
    ));
    let result = tools::execute(call, config, memory, run_id).await;
    let content = result.unwrap_or_else(|error| format!("tool failed: {error}"));
    let kind = if tool == "agent.finish" {
        "finish"
    } else if tool == "memory.write" {
        "memory_write"
    } else {
        "tool_result"
    };
    events.push(event(kind, content.clone(), Some(tool), Some(step)));
    content
}
