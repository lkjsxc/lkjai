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
                "<run id=\"{run_id}\" step=\"{step}\">\n<summary>\n{summary}\n</summary>\n<memories>\n{memories}\n</memories>\n<events>\n{}\n</events>\n</run>",
                event_tags(events)
            ),
        },
    ]
}

fn event_tags(events: &[Event]) -> String {
    events
        .iter()
        .rev()
        .take(20)
        .rev()
        .map(|event| {
            format!(
                "<event kind=\"{}\">{}</event>",
                event.kind,
                event.content.replace('&', "&amp;").replace('<', "&lt;")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn system_prompt() -> String {
    [
        "You are lkjai, a local agent. Read the tagged prompt sections.",
        "Return exactly one JSON object and no prose outside it.",
        r#"Use {"kind":"final","content":"..."} to answer."#,
        r#"Use {"kind":"tool_call","tool":"fs.read","args":{"path":"..."}}."#,
        r#"Use {"kind":"plan","content":"Search docs, then read the contract."} to outline steps before the first tool call."#,
        r#"Use {"kind":"request_confirmation","summary":"...","operation":"resource.update_resource","pending_tool_call":{"tool":"resource.update_resource","args":{"ref":"release-notes","body":"..."}}} for any create or update in kjxlkj."#,
        "Tools: shell.exec(command), web.fetch(url), fs.read(path), fs.write(path, content), fs.list(path), memory.search(query), memory.write(content), resource.search(query, kind), resource.fetch(ref), resource.history(ref), resource.preview_markdown(body, current_resource_id), resource.create_note(body, alias, is_private), resource.update_resource(ref, body, alias, is_favorite, is_private).",
        "Writes in kjxlkj require confirmation first. Search, fetch, history, and preview can run directly.",
    ]
    .join("\n")
}
