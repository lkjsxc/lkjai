use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{action::Action, event, Event};

const MUTATIONS: &[&str] = &[
    "resource.create_note",
    "resource.create_media",
    "resource.update_resource",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Pending {
    pub summary: String,
    pub operation: String,
    pub pending_tool: String,
    pub fields: BTreeMap<String, String>,
}

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
    if !MUTATIONS.contains(&pending_tool.as_str()) {
        return Err(format!("confirmation pending_tool must be a mutation: {pending_tool}"));
    }
    let pending = Pending {
        summary: summary.clone(),
        operation,
        pending_tool: pending_tool.clone(),
        fields: action.fields(),
    };
    events.push(event(
        "confirmation_request",
        serde_json::to_string(&pending).map_err(|error| error.to_string())?,
        Some(pending_tool),
        Some(step),
    ));
    Ok(summary)
}

pub fn pending(events: &[Event]) -> Option<Pending> {
    let mut found = None;
    for event in events {
        if event.kind == "confirmation_request" {
            found = serde_json::from_str::<Pending>(&event.content).ok();
        }
        if found.is_some() && event.kind == "user" {
            found = None;
        }
    }
    found
}

pub fn confirmed(text: &str) -> bool {
    matches!(text.trim().to_ascii_lowercase().as_str(), "yes" | "y" | "ok" | "proceed" | "confirm")
}

pub fn cancelled(text: &str) -> bool {
    matches!(text.trim().to_ascii_lowercase().as_str(), "no" | "n" | "cancel" | "stop")
}

pub fn is_mutation(tool: &str) -> bool {
    MUTATIONS.contains(&tool)
}
