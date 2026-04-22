import json
from pathlib import Path


def evaluate_fixed_suite(paths, threshold: float = 0.8) -> Path:
    cases = [
        case("fixtures-exist", paths.fixtures.exists(), str(paths.fixtures)),
        case("dataset-metadata-exists", paths.dataset_metadata.exists(), str(paths.dataset_metadata)),
        case("adapter-manifest-exists", paths.adapter_manifest.exists(), str(paths.adapter_manifest)),
        case("export-manifest-exists", paths.export_manifest.exists(), str(paths.export_manifest)),
        case("tool-trajectory-present", contains(paths.fixtures, "tool_trajectory"), "tool_trajectory"),
        case("memory-case-present", contains(paths.fixtures, "memory.write"), "memory.write"),
    ]
    passed = sum(1 for item in cases if item["passed"])
    report = {
        "threshold": threshold,
        "pass_rate": passed / len(cases),
        "passed": passed,
        "total": len(cases),
        "cases": cases,
    }
    paths.runs.mkdir(parents=True, exist_ok=True)
    out = paths.runs / "fixed-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def case(case_id: str, passed: bool, detail: str) -> dict:
    return {"id": case_id, "passed": bool(passed), "detail": detail}


def contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8")
