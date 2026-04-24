use super::{action::Action, event, Event};

pub fn handle(action: Action, step: usize, events: &mut Vec<Event>) -> Result<String, String> {
    let summary = action
        .field("summary")
        .ok_or_else(|| "request_confirmation missing summary".to_string())?;
    let operation = action
        .field("operation")
        .ok_or_else(|| "request_confirmation missing operation".to_string())?;
    let pending_tool = action
        .field("pending_tool")
        .ok_or_else(|| "request_confirmation missing pending_tool".to_string())?;
    events.push(event(
        "confirmation_request",
        format!("{summary}\noperation={operation}\npending_tool={pending_tool}"),
        Some(pending_tool),
        Some(step),
    ));
    Ok(summary)
}
