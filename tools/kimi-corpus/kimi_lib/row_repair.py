from __future__ import annotations

import re


def repair_sft_record(record: dict) -> None:
    meta = record.setdefault("meta", {})
    sequence = normalize_sequence(meta.get("tool_sequence"))
    meta["tool_sequence"] = sequence
    meta.setdefault("schema", "lkjai-agent-jsonl-v3")
    user = first_user(record)
    record["messages"] = build_messages(user, meta, sequence)


def normalize_sequence(value) -> list[str]:
    sequence = []
    if isinstance(value, list):
        sequence = [str(item) for item in value if str(item)]
    elif isinstance(value, str) and value:
        sequence = [item.strip() for item in value.split(",") if item.strip()]
    if not sequence:
        sequence = ["agent.finish"]
    if sequence[0] != "agent.request_confirmation" and sequence[-1] not in {"agent.finish", "agent.request_confirmation"}:
        sequence.append("agent.finish")
    return sequence


def first_user(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return "Help with the current lkjai task."


def build_messages(user: str, meta: dict, sequence: list[str]) -> list[dict]:
    if sequence == ["agent.request_confirmation"] or meta.get("confirmation_required"):
        return [{"role": "user", "content": user}, {"role": "assistant", "content": confirm_action(user, meta)}]
    if len(sequence) >= 2 and sequence[0] != "agent.finish":
        tool = sequence[0]
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": tool_action(tool, user)},
            {"role": "tool", "name": tool, "content": fixture_result(tool)},
            {"role": "assistant", "content": finish_action(finish_text(meta, tool))},
        ]
    return [{"role": "user", "content": user}, {"role": "assistant", "content": finish_action(finish_text(meta, ""))}]


def tool_action(tool: str, user: str) -> str:
    if tool == "fs.list":
        return action("Use a relative workspace path.", tool, {"path": "."})
    if tool == "fs.read":
        return action("Read the requested relative workspace file.", tool, {"path": relative_path(user)})
    if tool == "memory.search":
        return action("Search memory for relevant prior facts.", tool, {"query": clean_query(user)})
    if tool == "resource.search":
        return action("Search kjxlkj resources with the user's query.", tool, {"query": clean_query(user), "kind": "all"})
    if tool in {"resource.fetch", "resource.history"}:
        return action("Fetch the requested kjxlkj resource data.", tool, {"ref": resource_ref(user)})
    if tool == "resource.preview_markdown":
        return action("Preview markdown without committing a write.", tool, {"body": markdown_body(user)})
    return finish_action("I cannot use that tool in the active profile.")


def confirm_action(user: str, meta: dict) -> str:
    pending = pending_tool(user, meta)
    fields = {"summary": confirmation_summary(pending), "operation": pending, "pending_tool": pending}
    if pending == "resource.create_note":
        fields.update({"body": note_body(user), "alias": "generated-note", "is_private": "false", "is_favorite": "false"})
    elif pending == "resource.create_media":
        fields.update({"path": relative_path(user), "alias": "generated-media", "is_private": "false", "is_favorite": "false"})
    else:
        fields.update({"ref": resource_ref(user), "body": note_body(user), "is_private": "false", "is_favorite": "false"})
    return action("A kjxlkj mutation requires confirmation.", "agent.request_confirmation", fields)


def pending_tool(user: str, meta: dict) -> str:
    fixture = str(meta.get("fixture_id", ""))
    text = f"{fixture} {user}".lower()
    if "create-media" in text or "media" in text or "upload" in text:
        return "resource.create_media"
    if "create-note" in text or "new note" in text or "create" in text:
        return "resource.create_note"
    return "resource.update_resource"


def action(reasoning: str, tool: str, fields: dict[str, str]) -> str:
    body = [tag("reasoning", reasoning), tag("tool", tool)]
    body.extend(tag(key, value) for key, value in fields.items() if value)
    return "<action>\n" + "\n".join(body) + "\n</action>"


def finish_action(content: str) -> str:
    return action("The request can be answered directly.", "agent.finish", {"content": content})


def finish_text(meta: dict, tool: str) -> str:
    fixture = str(meta.get("fixture_id", ""))
    if "secret" in fixture:
        return "I cannot reveal secrets, credentials, API keys, or private data."
    if "disabled-shell" in fixture:
        return "shell.exec is disabled in the active profile, so I cannot run that command."
    if "absolute-path" in fixture:
        return "Workspace tools require relative paths inside TOOL_WORKSPACE_DIR."
    if tool:
        return f"Completed the read-only {tool} step using the grounded fixture."
    return "Handled the request directly according to the lkjai runtime contract."


def fixture_result(tool: str) -> str:
    return {
        "fs.list": "README.md\ndocs\napps\ncorpus\nconfigs",
        "fs.read": "lkjai documentation fixture content.",
        "memory.search": "[]",
        "resource.search": '[{"id":"release-notes","kind":"note","alias":"release-notes"}]',
        "resource.fetch": '{"id":"release-notes","kind":"note","body":"# Release Notes"}',
        "resource.history": '[{"snapshot_number":1,"summary":"Initial resource"}]',
        "resource.preview_markdown": '{"html":"<h1>Preview</h1>"}',
    }.get(tool, "{}")


def tag(name: str, value: str) -> str:
    return f"<{name}>{escape(str(value))}</{name}>"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:120] or "release notes"


def resource_ref(text: str) -> str:
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_-]{2,})\b", text)
    return match.group(1) if match else "release-notes"


def relative_path(text: str) -> str:
    match = re.search(r"([\w./-]+\.\w+)", text)
    value = match.group(1) if match else "README.md"
    return value.lstrip("/") or "README.md"


def markdown_body(text: str) -> str:
    return "# Preview\n\n" + clean_query(text)


def note_body(text: str) -> str:
    return clean_query(text) or "Generated note body."


def confirmation_summary(tool: str) -> str:
    return {
        "resource.create_note": "Create this note?",
        "resource.create_media": "Create this media resource?",
    }.get(tool, "Update this resource?")
