import json
import re

from .formatting import load_rows
from .dataset import parse_assistant_xml


STOPWORDS = {"the", "this", "that", "with", "from", "into", "about", "keep", "must", "will", "have", "uses", "after", "only"}


def evaluate_behavior(paths, settings, threshold: float = 0.6):
    from .generation import LoadedModel

    model = LoadedModel(paths.root.parent / "models" / settings.model_name, device="cpu")
    rows = [row for row in load_rows(paths.holdout_dataset) if row["messages"][-1]["role"] == "assistant"][:200]
    cases = [run_case(model, row) for row in rows]
    passed = sum(1 for item in cases if item["passed"])
    valid_xml = sum(1 for item in cases if item["valid_xml"])
    report = {
        "threshold": threshold,
        "pass_rate": passed / max(1, len(cases)),
        "xml_validity": valid_xml / max(1, len(cases)),
        "passed": passed,
        "total": len(cases),
        "valid_xml": valid_xml,
        "buckets": bucket_rates(cases),
        "cases": cases,
    }
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "behavioral-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def run_case(model, row: dict) -> dict:
    messages = row["messages"][:-1]
    expected = parse_assistant_xml(row["messages"][-1]["content"])
    text = model.complete(messages, max_tokens=96, temperature=0.0)
    try:
        actual = parse_assistant_xml(text)
    except ValueError as error:
        return result(row, False, False, f"invalid xml: {error}", text)
    schema_error = action_schema_error(actual)
    if schema_error:
        return result(row, False, False, schema_error, text)
    passed = compare_actions(expected, actual)
    return result(row, passed, True, json.dumps(actual, sort_keys=True), text)


def action_schema_error(action: dict) -> str:
    tool = action.get("tool")
    if not tool:
        return "action missing tool"
    if tool == "agent.finish" and not isinstance(action.get("content"), str):
        return "agent.finish missing content"
    if tool == "agent.request_confirmation" and not action.get("pending_tool"):
        return "request_confirmation missing pending tool"
    return ""


def compare_actions(expected: dict, actual: dict) -> bool:
    if expected.get("tool") != actual.get("tool"):
        return False
    tool = expected.get("tool")
    if tool == "agent.finish":
        return content_match(str(expected.get("content", "")), str(actual.get("content", "")))
    if tool == "agent.request_confirmation":
        return expected.get("pending_tool") == actual.get("pending_tool")
    comparable = {k: v for k, v in expected.items() if k not in {"reasoning"}}
    actual_subset = {k: actual.get(k) for k in comparable}
    return comparable == actual_subset


def content_match(expected: str, actual: str) -> bool:
    expected_lower, actual_lower = expected.lower(), actual.lower()
    if expected_lower in actual_lower or actual_lower in expected_lower:
        return True
    keywords = [word for word in re.findall(r"[a-z0-9_/-]{4,}", expected_lower) if word not in STOPWORDS]
    needed = min(3, len(set(keywords)))
    return len({word for word in keywords if word in actual_lower}) >= needed


def result(row: dict, passed: bool, valid_xml: bool, detail: str, output: str) -> dict:
    return {"id": row["meta"]["id"], "bucket": bucket(row), "passed": bool(passed), "valid_xml": bool(valid_xml), "detail": detail, "output": output[:500]}


def bucket(row: dict) -> str:
    tags = set(row.get("tags", []))
    if "agentic" in tags:
        if "revision" in tags:
            return "agentic_revision"
        if "tool_chain" in tags:
            return "agentic_tool_chain"
        if "planning" in tags:
            return "agentic_planning"
        return "agentic_multi_turn"
    if "confirmation" in tags:
        return "kjxlkj_mutation_confirmation"
    if "kjxlkj" in tags:
        return "kjxlkj_read_tool"
    if "workspace_tool" in tags or "runtime_tool" in tags:
        return "local_tool"
    if "safety" in tags:
        return "safety"
    if "docs_grounding" in tags:
        return "docs_grounding"
    return "direct_answer"


def bucket_rates(cases: list[dict]) -> dict:
    buckets: dict[str, dict] = {}
    for case in cases:
        item = buckets.setdefault(case["bucket"], {"passed": 0, "total": 0, "xml_valid": 0})
        item["passed"] += int(case["passed"])
        item["xml_valid"] += int(case["valid_xml"])
        item["total"] += 1
    return {
        name: {
            **item,
            "pass_rate": item["passed"] / max(1, item["total"]),
            "xml_validity": item["xml_valid"] / max(1, item["total"]),
        }
        for name, item in sorted(buckets.items())
    }
