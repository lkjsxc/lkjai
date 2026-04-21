import json
import math
from pathlib import Path


def evaluate_fixed_suite(paths, threshold: float = 0.8) -> Path:
    train = read_json(paths.runs / "last-train.json")
    size = read_json(paths.models / "lkj-150m" / "size.json")
    losses = [float(value) for value in train.get("losses", []) if finite(value)]
    steps = int(train.get("steps_completed", len(losses)))
    duration = float(train.get("duration_secs", 0.0))
    size_mib = float(size.get("size_mib", math.inf))
    cases = [
        case("loss-samples-at-least-8", len(losses) >= 8, f"loss_count={len(losses)}"),
        case("loss-decreases", loss_decreases(losses), loss_delta_detail(losses)),
        case("tail-loss-below-head", tail_below_head(losses), tail_ratio_detail(losses)),
        case("checkpoint-latest-exists", (paths.checkpoints / "latest.pt").exists(), "checkpoints/latest.pt"),
        case("export-under-512mib", size_mib <= 512.0, f"size_mib={size_mib:.4f}"),
        case("runtime-at-least-10m", duration >= 600.0, f"duration_secs={duration:.2f}"),
        case("steps-at-least-128", steps >= 128, f"steps_completed={steps}"),
    ]
    passed = sum(1 for item in cases if item["passed"])
    total = len(cases)
    report = {
        "threshold": threshold,
        "pass_rate": (passed / total) if total else 0.0,
        "passed": passed,
        "total": total,
        "cases": cases,
    }
    out = paths.runs / "fixed-eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def case(case_id: str, passed: bool, detail: str) -> dict:
    return {"id": case_id, "passed": passed, "detail": detail}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def loss_decreases(losses: list[float]) -> bool:
    return len(losses) >= 2 and losses[-1] < losses[0]


def loss_delta_detail(losses: list[float]) -> str:
    if len(losses) < 2:
        return "insufficient-loss-points"
    return f"first={losses[0]:.6f} final={losses[-1]:.6f}"


def tail_below_head(losses: list[float]) -> bool:
    if len(losses) < 6:
        return False
    thirds = max(1, len(losses) // 3)
    head = sum(losses[:thirds]) / thirds
    tail = sum(losses[-thirds:]) / thirds
    return tail < head * 0.95


def tail_ratio_detail(losses: list[float]) -> str:
    if len(losses) < 6:
        return "insufficient-loss-points"
    thirds = max(1, len(losses) // 3)
    head = sum(losses[:thirds]) / thirds
    tail = sum(losses[-thirds:]) / thirds
    ratio = tail / head if head else math.inf
    return f"tail_over_head={ratio:.6f}"
