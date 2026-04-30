use crate::config::Config;

use super::{confirmation, event, memory::MemoryStore, tool_runner, tools, Action, Event};

pub async fn respond(
    message: &str,
    prior: &[Event],
    config: &Config,
    memory: &MemoryStore,
    run_id: &str,
    events: &mut Vec<Event>,
) -> Option<(String, String)> {
    let pending = confirmation::pending(prior)?;
    if confirmation::confirmed(message) {
        let action = Action::new(pending.pending_tool.clone(), pending.fields);
        let call = match tools::ToolCall::from_fields(&action) {
            Ok(call) => call,
            Err(error) => return Some(finish(events, format!("Pending operation is invalid: {error}"))),
        };
        let result = tool_runner::run(call, config, memory, run_id, 1, events).await;
        events.push(event("observation", result, Some(pending.pending_tool), Some(1)));
        return Some(finish(events, "Confirmed and completed.".into()));
    }
    if confirmation::cancelled(message) {
        return Some(finish(events, "Cancelled pending operation.".into()));
    }
    Some(finish(events, "Cancelled pending operation before handling the new request.".into()))
}

fn finish(events: &mut Vec<Event>, content: String) -> (String, String) {
    events.push(event("finish", content.clone(), Some("agent.finish".into()), Some(1)));
    events.push(event("assistant", content.clone(), None, Some(1)));
    (content, "finish".into())
}
