import re

from .constants import DEFAULT_MODEL, FAMILY_SKILLS, MUTATION_TOOLS, SCHEMA, TOOLS
from .validate import action_tools_for_messages

TOOL_ALIASES = {
    "fs.read": "resource.get",
    "fs.list": "resource.search",
    "memory.search": "resource.search",
    "kjxlkj.read": "resource.get",
    "kjxlkj.fetch": "resource.get",
    "resource.fetch": "resource.get",
    "kjxlkj.list": "resource.search",
    "kjxlkj.search": "resource.search",
    "resource.create_note": "resource.create",
    "resource.create_media": "resource.create",
    "kjxlkj.create": "resource.create",
    "kjxlkj.update": "resource.update_resource",
    "resource.update": "resource.update_resource",
    "kjxlkj.delete": "resource.delete",
    "agent.error": "agent.finish",
}

TOOL_RE = re.compile(r"(<tool>\s*)([^<]+?)(\s*</tool>)")


def normalize_row(row, config, split, family, ordinal, offset):
    if not isinstance(row, dict):
        return row
    meta = dict(row.get("meta") or {})
    if isinstance(row.get("messages"), list):
        normalize_messages(row["messages"])
    action_tools = []
    if isinstance(row.get("messages"), list):
        action_tools = action_tools_for_messages(row["messages"], [])
    fixture_id = meta.get("fixture_id") or "repo-grounding"
    compaction = requires_compaction(config, ordinal)
    multi_turn = compaction or requires_multi_turn(config, ordinal)
    meta.update(
        {
            "schema": SCHEMA,
            "id": f"kimi-{split}-{ordinal:09d}-{offset:03d}",
            "split": split,
            "provenance": "kimi-generated",
            "author_type": "external-agent-generated",
            "author_model": str(config.get("api_model", DEFAULT_MODEL)),
            "quality_tier": meta.get("quality_tier", "high"),
            "toolset": "core",
            "language": "en",
            "safety_scope": "workspace-safe",
            "license": "project-local",
            "source_ref": meta.get("source_ref", f"corpus/fixtures/repo-grounding.json#{fixture_id}"),
            "mode": "sft",
            "prompt_contract": "agent-api",
            "template_family": family,
            "scenario_family_id": f"kimi-{family}-{split}-{ordinal:09d}-{offset:03d}",
            "tool_sequence": action_tools,
            "confirmation_required": "agent.request_confirmation" in action_tools,
            "grounding_source": "repo_fixture",
            "fixture_id": fixture_id,
            "contract_validated": True,
            "fixture_executed": True,
        }
    )
    if multi_turn:
        meta["multi_turn"] = True
    if compaction:
        meta["compaction"] = True
    meta.setdefault("domain", "lkjai-agent")
    meta.setdefault("skill", FAMILY_SKILLS.get(family, "agent-action"))
    meta.setdefault("intent", family)
    meta.setdefault(
        "gold_stop_reason",
        "confirmation_required" if meta["confirmation_required"] else "finish",
    )
    row["meta"] = meta
    row["tags"] = merged_tags(row.get("tags"), compaction)
    return row


def normalize_messages(messages):
    previous_tool = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                continue
            content, previous_tool = normalize_assistant_content(content)
            message["content"] = content
        elif role == "tool":
            name = message.get("name")
            if isinstance(name, str) and name.strip():
                message["name"] = normalized_tool_name(name)
            elif previous_tool in TOOLS:
                message["name"] = previous_tool


def normalize_assistant_content(content):
    if content.count("<action>") != 1 or content.count("</action>") != 1:
        return finish_action(content), "agent.finish"

    match = TOOL_RE.search(content)
    if not match:
        return finish_action(content), "agent.finish"

    raw_tool = match.group(2).strip()
    tool = normalized_tool_name(raw_tool)
    if tool in MUTATION_TOOLS:
        content = TOOL_RE.sub(r"\1agent.request_confirmation\3", content, count=1)
        if "pending_tool" not in content:
            insert_at = content.find("</action>")
            pending = f'\n<param name="pending_tool">{tool}</param>'
            content = content[:insert_at] + pending + content[insert_at:]
        return content, "agent.request_confirmation"
    if tool not in TOOLS:
        tool = "agent.finish"
    content = TOOL_RE.sub(lambda m: f"{m.group(1)}{tool}{m.group(3)}", content, count=1)
    return content, tool


def normalized_tool_name(tool):
    tool = str(tool).strip()
    return TOOL_ALIASES.get(tool, tool)


def finish_action(message):
    message = str(message).strip()
    if not message:
        message = "Done."
    return (
        "<action>\n"
        "<tool>agent.finish</tool>\n"
        f'<param name="message">{escape_xml_text(message)}</param>\n'
        "</action>"
    )


def escape_xml_text(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def requires_multi_turn(config, ordinal):
    return ordinal % 100 < int(float(config.get("multi_turn_ratio", 0.60)) * 100)


def requires_compaction(config, ordinal):
    return ordinal % 100 < int(float(config.get("compaction_ratio", 0.25)) * 100)


def merged_tags(tags, compaction=False):
    result = []
    if not isinstance(tags, list):
        tags = []
    extra = ["sft", "repo_fixture", "language:en"]
    if compaction:
        extra.append("compacted-context")
    for tag in [*tags, *extra]:
        if isinstance(tag, str) and tag not in result:
            result.append(tag)
    return result
