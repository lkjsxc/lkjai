use super::{action::Action, event, Event};

pub fn handle(action: Action, step: usize, events: &mut Vec<Event>) -> Result<String, String> {
    let summary = action
        .summary
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "request_confirmation missing summary".to_string())?;
    let operation = action
        .operation
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "request_confirmation missing operation".to_string())?;
    let pending = action
        .pending_tool_call
        .ok_or_else(|| "request_confirmation missing pending_tool_call".to_string())?;
    events.push(event(
        "confirmation_request",
        format!(
            "{summary}\noperation={operation}\npending_tool={}\npending_args={}",
            pending.tool, pending.args
        ),
        Some(pending.tool),
        Some(step),
    ));
    Ok(summary)
}
