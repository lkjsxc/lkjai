from __future__ import annotations

import re
from xml.etree import ElementTree


MUTATIONS = {"resource.create_note", "resource.create_media", "resource.update_resource"}
API_META = {
    "template_family",
    "scenario_family_id",
    "intent",
    "tool_sequence",
    "confirmation_required",
    "grounding_source",
    "gold_stop_reason",
}


def validate_agent_sft(row: dict) -> list[str]:
    meta = row.get("meta", {})
    flags: list[str] = []
    if str(meta.get("prompt_version", "")).startswith("api") and API_META - meta.keys():
        flags.append("missing_api_agent_meta")
    tools = assistant_tools(row)
    if tools and tools[-1] != "agent.finish":
        flags.append("last_assistant_not_finish")
    for index, tool in enumerate(tools):
        if tool in MUTATIONS and "agent.request_confirmation" not in tools[:index]:
            flags.append("mutation_without_confirmation")
            break
    sequence = meta.get("tool_sequence")
    if isinstance(sequence, list) and tools and sequence != tools:
        flags.append("tool_sequence_mismatch")
    return flags


def assistant_tools(row: dict) -> list[str]:
    tools = []
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        tool = xml_tool(str(message.get("content", "")))
        if tool:
            tools.append(tool)
    return tools


def xml_tool(text: str) -> str:
    match = re.search(r"<action>(.*)</action>\s*$", text.strip(), re.S)
    if not match:
        return ""
    try:
        root = ElementTree.fromstring("<action>" + match.group(1) + "</action>")
    except ElementTree.ParseError:
        return ""
    child = root.find("tool")
    return (child.text or "").strip() if child is not None else ""
