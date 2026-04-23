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
    passed = compare_actions(expected, actual)
    return result(row, passed, True, json.dumps(actual, sort_keys=True), text)


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
    return {"id": row["meta"]["id"], "passed": bool(passed), "valid_json": bool(valid_json), "detail": detail, "output": output[:500]}
