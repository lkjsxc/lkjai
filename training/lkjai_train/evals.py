import json
from pathlib import Path


def evaluate_fixed_suite(paths, threshold: float = 0.8) -> Path:
    summary = read_json(paths.training_summary) if paths.training_summary.exists() else {}
    cases = [
        case("fixtures-exist", paths.fixtures.exists(), str(paths.fixtures)),
        case("dataset-metadata-exists", paths.dataset_metadata.exists(), str(paths.dataset_metadata)),
        case("training-summary-exists", paths.training_summary.exists(), str(paths.training_summary)),
        case("adapter-manifest-exists", paths.adapter_manifest.exists(), str(paths.adapter_manifest)),
        case("adapter-final-exists", paths.adapter_final.exists(), str(paths.adapter_final)),
        case("export-manifest-exists", paths.export_manifest.exists(), str(paths.export_manifest)),
        case("tool-trajectory-present", contains(paths.fixtures, "tool_trajectory"), "tool_trajectory"),
        case("memory-case-present", contains(paths.fixtures, "memory.write"), "memory.write"),
        case("adapter-has-weights", has_adapter_weights(paths.adapter_final), "adapter weights"),
        case("summary-has-loss", has_loss_metrics(summary), "loss metrics"),
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


def has_adapter_weights(adapter_dir: Path) -> bool:
    if not adapter_dir.exists():
        return False
    files = [p.name for p in adapter_dir.iterdir() if p.is_file()]
    has_config = "adapter_config.json" in files
    has_weights = any(name.endswith(".safetensors") or name.endswith(".bin") for name in files)
    return has_config and has_weights


def has_loss_metrics(summary: dict) -> bool:
    metrics = summary.get("metrics", {})
    return bool(metrics) and any("loss" in str(k).lower() for k in metrics.keys())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
