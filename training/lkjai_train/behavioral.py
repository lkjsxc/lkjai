import json
import re

from .formatting import load_rows
from .generation import LoadedModel


STOPWORDS = {"the", "this", "that", "with", "from", "into", "about", "keep", "must", "will", "have", "uses", "after", "only"}


def evaluate_behavior(paths, settings, threshold: float = 0.6):
    model = LoadedModel(paths.root.parent / "models" / settings.model_name, device="cpu")
    rows = [row for row in load_rows(paths.holdout_dataset) if row["messages"][-1]["role"] == "assistant"][:200]
    cases = [run_case(model, row) for row in rows]
    passed = sum(1 for item in cases if item["passed"])
    valid_json = sum(1 for item in cases if item["valid_json"])
    report = {
        "threshold": threshold,
        "pass_rate": passed / max(1, len(cases)),
        "json_validity": valid_json / max(1, len(cases)),
        "passed": passed,
        "total": len(cases),
        "valid_json": valid_json,
        "buckets": bucket_rates(cases),
        "cases": cases,
    }
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "behavioral-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def run_case(model: LoadedModel, row: dict) -> dict:
    messages = row["messages"][:-1]
    expected = json.loads(row["messages"][-1]["content"])
    text = model.complete(messages, max_tokens=96, temperature=0.0)
    try:
        actual = json.loads(text)
    except json.JSONDecodeError as error:
        return result(row, False, False, f"invalid json: {error}", text)
    schema_error = action_schema_error(actual)
    if schema_error:
        return result(row, False, False, schema_error, text)
    passed = compare_actions(expected, actual)
    return result(row, passed, True, json.dumps(actual, sort_keys=True), text)


def action_schema_error(action: dict) -> str:
    kind = action.get("kind")
    if kind == "final":
        return "" if isinstance(action.get("content"), str) else "final missing string content"
    if kind == "tool_call":
        if not isinstance(action.get("tool"), str):
            return "tool_call missing string tool"
        return "" if isinstance(action.get("args", {}), dict) else "tool_call args must be object"
    if kind == "request_confirmation":
        pending = action.get("pending_tool_call")
        if not isinstance(action.get("operation"), str):
            return "request_confirmation missing string operation"
        if not isinstance(pending, dict) or not isinstance(pending.get("tool"), str):
            return "request_confirmation missing pending tool"
        return "" if isinstance(pending.get("args", {}), dict) else "pending args must be object"
    return f"unknown action kind {kind}"


def compare_actions(expected: dict, actual: dict) -> bool:
    if expected.get("kind") != actual.get("kind"):
        return False
    if expected["kind"] == "tool_call":
        return expected.get("tool") == actual.get("tool") and expected.get("args", {}) == actual.get("args", {})
    if expected["kind"] == "request_confirmation":
        pending = actual.get("pending_tool_call", {})
        expected_pending = expected.get("pending_tool_call", {})
        return expected.get("operation") == actual.get("operation") and pending.get("tool") == expected_pending.get("tool")
    return content_match(str(expected.get("content", "")), str(actual.get("content", "")))


def content_match(expected: str, actual: str) -> bool:
    expected_lower, actual_lower = expected.lower(), actual.lower()
    if expected_lower in actual_lower or actual_lower in expected_lower:
        return True
    keywords = [word for word in re.findall(r"[a-z0-9_/-]{4,}", expected_lower) if word not in STOPWORDS]
    needed = min(3, len(set(keywords)))
    return len({word for word in keywords if word in actual_lower}) >= needed


def result(row: dict, passed: bool, valid_json: bool, detail: str, output: str) -> dict:
    return {"id": row["meta"]["id"], "bucket": bucket(row), "passed": bool(passed), "valid_json": bool(valid_json), "detail": detail, "output": output[:500]}


def bucket(row: dict) -> str:
    tags = set(row.get("tags", []))
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
        item = buckets.setdefault(case["bucket"], {"passed": 0, "total": 0, "json_valid": 0})
        item["passed"] += int(case["passed"])
        item["json_valid"] += int(case["valid_json"])
        item["total"] += 1
    return {
        name: {
            **item,
            "pass_rate": item["passed"] / max(1, item["total"]),
            "json_validity": item["json_valid"] / max(1, item["total"]),
        }
        for name, item in sorted(buckets.items())
    }
