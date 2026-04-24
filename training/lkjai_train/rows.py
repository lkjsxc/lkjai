import json
from html import escape


def action_message(action: dict) -> dict:
    return {"role": "assistant", "content": action_xml(action)}


def action_xml(action: dict) -> str:
    kind = action.get("kind")
    if kind == "final":
        return xml_action("agent.finish", {"content": action.get("content", "")}, action.get("thought", ""))
    if kind == "tool_call":
        return xml_action(action.get("tool", ""), action.get("args", {}), action.get("thought", ""))
    if kind == "request_confirmation":
        pending = action.get("pending_tool_call", {})
        fields = {
            "summary": action.get("summary", ""),
            "operation": action.get("operation", ""),
            "pending_tool": pending.get("tool", ""),
            **pending.get("args", {}),
        }
        return xml_action("agent.request_confirmation", fields, action.get("thought", ""))
    if kind == "plan":
        return xml_action("agent.think", {"content": action.get("content", "")}, action.get("thought", ""))
    return xml_action(str(kind or ""), {}, action.get("thought", ""))


def xml_action(tool: str, fields: dict, reasoning: str = "") -> str:
    lines = ["<action>"]
    if reasoning:
        lines.append(tag("reasoning", reasoning))
    lines.append(tag("tool", tool))
    for key, value in fields.items():
        if value is None:
            continue
        lines.append(tag(str(key), value))
    lines.append("</action>")
    return "\n".join(lines)


def tag(name: str, value) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return f"<{name}>{escape(text, quote=False)}</{name}>"


def meta(
    row_id: str,
    domain: str,
    skill: str,
    source_ref: str,
    *,
    split: str,
    toolset: str = "none",
    language: str = "en",
    license_name: str = "project-local",
    safety_scope: str = "workspace-safe",
) -> dict:
    return {
        "id": row_id,
        "split": split,
        "provenance": "repo-derived",
        "author_type": "repo-derived",
        "author_model": "none",
        "quality_tier": "high",
        "domain": domain,
        "skill": skill,
        "toolset": toolset,
        "language": language,
        "safety_scope": safety_scope,
        "license": license_name,
        "source_ref": source_ref,
    }


def kimi_meta(
    row_id: str,
    domain: str,
    skill: str,
    source_ref: str,
    *,
    split: str,
    toolset: str = "none",
    language: str = "en",
    license_name: str = "project-local",
    safety_scope: str = "workspace-safe",
) -> dict:
    return {
        "id": row_id,
        "split": split,
        "provenance": "kimi-generated",
        "author_type": "external-agent-generated",
        "author_model": "kimi-code",
        "quality_tier": "high",
        "domain": domain,
        "skill": skill,
        "toolset": toolset,
        "language": language,
        "safety_scope": safety_scope,
        "license": license_name,
        "source_ref": source_ref,
    }


def row(messages: list[dict], tags: list[str], metadata: dict) -> dict:
    return {"messages": messages, "tags": sorted(set(tags)), "meta": metadata}


def signature(row: dict) -> str:
    payload = {
        "messages": row.get("messages", []),
        "tags": sorted(set(row.get("tags", []))),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def direct_row(prompt: str, answer: str, tags: list[str], metadata: dict) -> dict:
    return row(
        [{"role": "user", "content": prompt}, action_message({"kind": "final", "content": answer})],
        tags,
        metadata,
    )


def tool_only_row(
    prompt: str,
    tool: str,
    args: dict,
    tags: list[str],
    metadata: dict,
    thought: str = "use tool",
) -> dict:
    return row(
        [
            {"role": "user", "content": prompt},
            action_message({"kind": "tool_call", "thought": thought, "tool": tool, "args": args}),
        ],
        tags,
        metadata,
    )


def tool_row(
    prompt: str,
    tool: str,
    args: dict,
    result: str,
    final_answer: str,
    tags: list[str],
    metadata: dict,
) -> dict:
    return row(
        [
            {"role": "user", "content": prompt},
            action_message({"kind": "tool_call", "tool": tool, "args": args}),
            {"role": "tool", "name": tool, "content": result},
            action_message({"kind": "final", "content": final_answer}),
        ],
        tags,
        metadata,
    )


def confirm_row(
    prompt: str,
    operation: str,
    pending_tool: str,
    pending_args: dict,
    summary: str,
    tags: list[str],
    metadata: dict,
) -> dict:
    return row(
        [
            {"role": "user", "content": prompt},
            action_message(
                {
                    "kind": "request_confirmation",
                    "summary": summary,
                    "operation": operation,
                    "pending_tool_call": {"tool": pending_tool, "args": pending_args},
                }
            ),
        ],
        tags,
        metadata,
    )


def plan_row(prompt: str, plan: str, tags: list[str], metadata: dict) -> dict:
    return row([{"role": "user", "content": prompt}, action_message({"kind": "plan", "content": plan})], tags, metadata)


def multi_turn_row(messages: list[dict], tags: list[str], metadata: dict) -> dict:
    return row(messages, tags, metadata)


def revise_row(
    prompt: str,
    plan: str,
    first_tool: str,
    first_args: dict,
    first_result: str,
    second_tool: str,
    second_args: dict,
    second_result: str,
    final_answer: str,
    tags: list[str],
    metadata: dict,
) -> dict:
    return row(
        [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": plan}),
            action_message({"kind": "tool_call", "tool": first_tool, "args": first_args}),
            {"role": "tool", "name": first_tool, "content": first_result},
            action_message({"kind": "tool_call", "tool": second_tool, "args": second_args}),
            {"role": "tool", "name": second_tool, "content": second_result},
            action_message({"kind": "final", "content": final_answer}),
        ],
        tags,
        metadata,
    )
