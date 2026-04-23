import json
from pathlib import Path

from .generation import LoadedModel


CASES = [
    {
        "id": "direct-hello",
        "messages": [{"role": "user", "content": "Say hello."}],
        "kind": "final",
        "contains": "hello",
    },
    {
        "id": "direct-math",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "kind": "final",
        "contains": "4",
    },
    {
        "id": "tool-list",
        "messages": [{"role": "user", "content": "List the files in the current directory."}],
        "kind": "tool_call",
        "tool": "fs.list",
    },
    {
        "id": "memory-write",
        "messages": [{"role": "user", "content": "Remember that I prefer concise answers."}],
        "kind": "tool_call",
        "tool": "memory.write",
    },
    {
        "id": "docs-grounding",
        "messages": [{"role": "user", "content": "What is lkjai?"}],
        "kind": "final",
        "contains": "scratch",
    },
]


def evaluate_behavior(paths, settings, threshold: float = 0.8) -> Path:
    model = LoadedModel(paths.root.parent / "models" / settings.model_name, device="cpu")
    cases = [run_case(model, item) for item in CASES]
    passed = sum(1 for item in cases if item["passed"])
    report = {
        "threshold": threshold,
        "pass_rate": passed / len(cases),
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "behavioral-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def run_case(model: LoadedModel, case: dict) -> dict:
    text = model.complete(case["messages"], max_tokens=96, temperature=0.0)
    try:
        action = json.loads(text)
    except json.JSONDecodeError as error:
        return result(case, False, f"invalid json: {error}", text)
    passed = action.get("kind") == case["kind"]
    if "tool" in case:
        passed = passed and action.get("tool") == case["tool"]
    if "contains" in case:
        content = str(action.get("content", "")).lower()
        passed = passed and case["contains"].lower() in content
    return result(case, passed, json.dumps(action, sort_keys=True), text)


def result(case: dict, passed: bool, detail: str, output: str) -> dict:
    return {
        "id": case["id"],
        "passed": bool(passed),
        "detail": detail,
        "output": output[:500],
    }
