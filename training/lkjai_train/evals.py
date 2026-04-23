import json
from pathlib import Path


def evaluate_fixed_suite(paths, threshold: float = 0.8) -> Path:
    summary = read_json(paths.training_summary) if paths.training_summary.exists() else {}
    dataset = paths.corpus if paths.corpus.exists() else paths.fixtures
    cases = [
        case("fixtures-exist", paths.fixtures.exists(), str(paths.fixtures)),
        case("dataset-metadata-exists", paths.dataset_metadata.exists(), str(paths.dataset_metadata)),
        case("training-summary-exists", paths.training_summary.exists(), str(paths.training_summary)),
        case("tokenizer-manifest-exists", paths.tokenizer_manifest.exists(), str(paths.tokenizer_manifest)),
        case("checkpoint-manifest-exists", paths.checkpoint_manifest.exists(), str(paths.checkpoint_manifest)),
        case("export-manifest-exists", paths.export_manifest.exists(), str(paths.export_manifest)),
        case("tool-trajectory-present", contains(dataset, "tool_trajectory"), "tool_trajectory"),
        case("memory-case-present", contains(dataset, "memory.write"), "memory.write"),
        case("checkpoint-has-weights", has_checkpoint_weights(paths.checkpoint_final), "checkpoint weights"),
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


def has_checkpoint_weights(checkpoint_dir: Path) -> bool:
    if not checkpoint_dir.exists():
        return False
    files = {p.name for p in checkpoint_dir.iterdir() if p.is_file()}
    return "config.json" in files and "model.pt" in files


def has_loss_metrics(summary: dict) -> bool:
    metrics = summary.get("metrics", {})
    return bool(metrics) and any("loss" in str(key).lower() for key in metrics.keys())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
