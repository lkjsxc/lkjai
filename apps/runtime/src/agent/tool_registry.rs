const READONLY: &[&str] = &[
    "agent.finish",
    "agent.think",
    "fs.read",
    "fs.list",
    "memory.search",
    "resource.search",
    "resource.fetch",
    "resource.history",
    "resource.preview_markdown",
];

const MUTABLE_EXTRA: &[&str] = &[
    "agent.request_confirmation",
    "resource.create_note",
    "resource.create_media",
    "resource.update_resource",
];

pub fn require_enabled(tool: &str, profile: &str) -> Result<(), String> {
    if enabled(tool, profile) {
        Ok(())
    } else {
        Err(format!("tool disabled in {profile} profile: {tool}"))
    }
}

pub fn require_trainable(tool: &str) -> Result<(), String> {
    if READONLY.contains(&tool) || MUTABLE_EXTRA.contains(&tool) {
        Ok(())
    } else {
        Err(format!("tool disabled for active corpus: {tool}"))
    }
}

pub fn tool_list(profile: &str) -> &'static str {
    if profile == "mutable" {
        "agent.think(content), agent.finish(content), agent.request_confirmation(summary, operation, pending_tool, fields), fs.read(path), fs.list(path), memory.search(query), resource.search(query, kind), resource.fetch(ref), resource.history(ref), resource.preview_markdown(body, current_resource_id), resource.create_note(body, alias, is_favorite, is_private), resource.create_media(path, alias, is_favorite, is_private), resource.update_resource(ref, body, alias, is_favorite, is_private)."
    } else {
        "agent.think(content), agent.finish(content), fs.read(path), fs.list(path), memory.search(query), resource.search(query, kind), resource.fetch(ref), resource.history(ref), resource.preview_markdown(body, current_resource_id)."
    }
}

fn enabled(tool: &str, profile: &str) -> bool {
    READONLY.contains(&tool) || profile == "mutable" && MUTABLE_EXTRA.contains(&tool)
}
