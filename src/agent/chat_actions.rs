use super::{confirmation, event, tools, Event};

pub(super) fn model_stop_reason(error: &str) -> String {
    if error.starts_with("model request failed")
        || error.starts_with("model server returned")
        || error.starts_with("model response parse failed")
    {
        "model_error".into()
    } else {
        "invalid_action".into()
    }
}

pub(super) fn finish_action(
    action: super::Action,
    step: usize,
    events: &mut Vec<Event>,
    assistant: &mut String,
    stop_reason: &mut String,
) -> bool {
    match tools::ToolCall::from_fields(&action) {
        Ok(tools::ToolCall::AgentFinish { content }) => {
            events.push(event(
                "finish",
                content.clone(),
                Some("agent.finish".into()),
                Some(step),
            ));
            *assistant = content.clone();
            events.push(event("assistant", content, None, Some(step)));
            *stop_reason = "finish".into();
        }
        Err(error) => {
            events.push(event("error", error, None, Some(step)));
            *stop_reason = "invalid_action".into();
        }
        _ => unreachable!(),
    }
    true
}

pub(super) fn think_action(
    action: super::Action,
    step: usize,
    max_steps: usize,
    base_prior: &[Event],
    prior: &mut Vec<Event>,
    events: &mut Vec<Event>,
    stop_reason: &mut String,
) -> bool {
    match tools::ToolCall::from_fields(&action) {
        Ok(tools::ToolCall::AgentThink { content }) => {
            events.push(event(
                "plan",
                content,
                Some("agent.think".into()),
                Some(step),
            ));
            *prior = base_prior.to_vec();
            prior.extend(events.clone());
            if step == max_steps {
                *stop_reason = "max_steps".into();
            }
            false
        }
        Err(error) => {
            events.push(event("error", error, None, Some(step)));
            *stop_reason = "invalid_action".into();
            true
        }
        _ => unreachable!(),
    }
}

pub(super) fn confirm_action(
    action: super::Action,
    step: usize,
    events: &mut Vec<Event>,
    assistant: &mut String,
    stop_reason: &mut String,
) -> bool {
    match confirmation::handle(action, step, events) {
        Ok(message) => {
            *assistant = message;
            *stop_reason = "confirmation_required".into();
        }
        Err(error) => {
            events.push(event("error", error, None, Some(step)));
            *stop_reason = "invalid_action".into();
        }
    }
    true
}
