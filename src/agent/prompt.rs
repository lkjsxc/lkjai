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
                "run_id={run_id}\nstep={step}\nsummary:\n{summary}\nmemories:\n{memories}\nrecent_events:\n{}",
                compact_events(events)
            ),
        },
    ]
}

fn compact_events(events: &[Event]) -> String {
    events
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|event| format!("{}: {}", event.kind, event.content.replace('\n', "\\n")))
        .collect::<Vec<_>>()
        .join("\n")
}

fn system_prompt() -> String {
    [
        "You are lkjai, a local agent. Return exactly one JSON object.",
        r#"Use {"kind":"final","thought":"...","content":"..."} to answer."#,
        r#"Use {"kind":"tool_call","thought":"...","tool":"fs.read","args":{"path":"..."}}."#,
        "Tools: shell.exec(command), web.fetch(url), fs.read(path), fs.write(path, content), fs.list(path), memory.search(query), memory.write(content).",
    ]
    .join("\n")
}
