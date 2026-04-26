use crate::model_client::ModelMessage;

use super::{memory::MemoryStore, Event};

pub fn build(
    run_id: &str,
    events: &[Event],
    step: usize,
    memory: &MemoryStore,
) -> Vec<ModelMessage> {
    let latest = events
        .last()
        .map(|event| event.content.as_str())
        .unwrap_or("");
    let memories = memory.search(latest, 5).unwrap_or_default().join("\n");
    let summary = memory.summary(run_id).ok().flatten().unwrap_or_default();
    vec![
        ModelMessage {
            role: "system".into(),
            content: system_prompt(),
        },
        ModelMessage {
            role: "user".into(),
            content: format!(
                "<run>\n<run_id>{run_id}</run_id>\n<step>{step}</step>\n<summary>\n{summary}\n</summary>\n<memories>\n{memories}\n</memories>\n<events>\n{}\n</events>\n</run>",
                event_tags(events)
            ),
        },
    ]
}

fn event_tags(events: &[Event]) -> String {
    compact_events(events)
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|event| {
            format!(
                "<event>\n<kind>{}</kind>\n<content>{}</content>\n</event>",
                event.kind,
                event.content.replace('&', "&amp;").replace('<', "&lt;")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compact_events(events: &[Event]) -> Vec<Event> {
    let mut compacted = Vec::new();
    let mut last_tool_result = "";
    for event in events {
        if event.kind == "observation" && event.content == last_tool_result {
            continue;
        }
        if event.kind == "tool_result" {
            last_tool_result = &event.content;
        }
        compacted.push(event.clone());
    }
    compacted
}

fn system_prompt() -> String {
    include_str!("../../prompts/codex-40m-system.txt")
        .trim()
        .to_string()
}
