from pathlib import Path
import re

from .common import estimated_tokens_for_row, iter_jsonl, write_json
from .constants import CORPUS, MUTATION_TOOLS, ROLES, SCHEMA, SPLITS, TOOLS

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def action_tools_for_messages(messages, errors):
    action_tools = []
    for target in [m for m in messages if m.get("role") == "assistant"]:
        content = target.get("content", "")
        if not isinstance(content, str):
            errors.append("assistant content must be string")
            continue
        if content.count("<action>") != 1 or content.count("</action>") != 1:
            errors.append("assistant target must contain one action")
            continue
        if content.find("<action>") > content.find("</action>"):
            errors.append("assistant action XML is malformed")
            continue
        if content.count("<tool>") != 1 or content.count("</tool>") != 1:
            errors.append("assistant action must contain one tool")
            continue
        tool = content.split("<tool>", 1)[1].split("</tool>", 1)[0].strip()
        action_tools.append(tool)
        if tool not in TOOLS:
            errors.append("tool not allowed")
        if tool in MUTATION_TOOLS:
            errors.append("mutation tool must be requested through confirmation")
    return action_tools


def validate_row(row, split, seen):
    errors = []
    if not isinstance(row, dict):
        return ["row must be an object"]
    meta = row.get("meta", {})
    messages = row.get("messages", [])
    tags = row.get("tags", [])
    if not isinstance(meta, dict):
        errors.append("meta must be object")
        meta = {}
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be non-empty list")
        messages = []
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append("tags must be string list")
    if meta.get("schema") != SCHEMA:
        errors.append("bad schema")
    if meta.get("split") != split:
        errors.append("split mismatch")
    if meta.get("provenance") != "kimi-generated":
        errors.append("bad provenance")
    if meta.get("mode") != "sft":
        errors.append("bad mode")
    if meta.get("prompt_contract") != "agent-api":
        errors.append("bad prompt contract")
    row_id = meta.get("id")
    if not isinstance(row_id, str) or not row_id:
        errors.append("missing id")
    elif row_id in seen:
        errors.append("duplicate id")
    else:
        seen.add(row_id)
    validate_messages(messages, errors)
    validate_english_only(messages, errors)
    action_tools = action_tools_for_messages(messages, errors)
    if not action_tools:
        errors.append("missing assistant action target")
    if meta.get("confirmation_required") and "agent.request_confirmation" not in action_tools:
        errors.append("confirmation row must request confirmation")
    if "agent.request_confirmation" in action_tools and not meta.get("confirmation_required"):
        errors.append("confirmation metadata missing")
    if isinstance(meta.get("tool_sequence"), list) and meta.get("tool_sequence") != action_tools:
        errors.append("tool sequence mismatch")
    if not meta.get("scenario_family_id"):
        errors.append("missing scenario family")
    if meta.get("multi_turn"):
        validate_multi_turn(messages, errors)
    if meta.get("compaction"):
        validate_compaction(messages, tags, errors)
    return errors


def validate_messages(messages, errors):
    for message in messages:
        if not isinstance(message, dict):
            errors.append("message must be object")
            continue
        role = message.get("role")
        if role not in ROLES:
            errors.append("role not allowed")
        if role == "tool":
            name = message.get("name")
            if not name:
                errors.append("tool message missing name")
            elif name not in TOOLS:
                errors.append("tool message name not allowed")
        if not isinstance(message.get("content"), str):
            errors.append("message content must be string")


def validate_english_only(messages, errors):
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, str) and CJK_RE.search(content):
            errors.append("messages must be English-only")
            return


def validate_multi_turn(messages, errors):
    objects = [message for message in messages if isinstance(message, dict)]
    user_turns = sum(1 for message in objects if message.get("role") == "user")
    assistant_actions = sum(1 for message in objects if message.get("role") == "assistant")
    tool_outputs = sum(1 for message in objects if message.get("role") == "tool")
    if user_turns < 2:
        errors.append("multi-turn row requires at least two user turns")
    if assistant_actions < 2:
        errors.append("multi-turn row requires at least two assistant actions")
    if tool_outputs < 1:
        errors.append("multi-turn row requires at least one tool output")


def validate_compaction(messages, tags, errors):
    if "compacted-context" not in tags:
        errors.append("compaction row missing compacted-context tag")
    if not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        errors.append("compaction row must start with system summary")
        return
    content = messages[0].get("content", "")
    if not isinstance(content, str) or not content.startswith("Conversation summary: "):
        errors.append("compaction summary must start with Conversation summary: ")


def validate(root, write_report=False):
    seen = set()
    family_splits = {}
    errors = []
    count = 0
    multi_turn_rows = 0
    compaction_rows = 0
    split_rows = {split: 0 for split in SPLITS}
    token_estimate = 0
    for split, path, number, row, parse_error in iter_jsonl(root):
        if parse_error:
            errors.append(f"{path}:{number}: {parse_error}")
            continue
        row_errors = validate_row(row, split, seen)
        count += 1
        split_rows[split] += 1
        token_estimate += estimated_tokens_for_row(row)
        if row.get("meta", {}).get("multi_turn"):
            multi_turn_rows += 1
        if row.get("meta", {}).get("compaction"):
            compaction_rows += 1
        family = row.get("meta", {}).get("scenario_family_id")
        if family:
            other = family_splits.setdefault(family, split)
            if other != split:
                row_errors.append("scenario family split leakage")
        errors.extend(f"{path}:{number}: {error}" for error in row_errors)
    report = {
        "schema": "lkjai-kimi-sft-validation",
        "status": "pass" if not errors and count > 0 else "fail",
        "corpus": CORPUS,
        "rows": count,
        "split_rows": split_rows,
        "token_estimate": token_estimate,
        "multi_turn_rows": multi_turn_rows,
        "multi_turn_ratio": (multi_turn_rows / count) if count else 0,
        "compaction_rows": compaction_rows,
        "compaction_ratio": (compaction_rows / count) if count else 0,
        "errors": errors,
    }
    if write_report:
        write_json(Path(root) / "validation-report.json", report)
    return report
