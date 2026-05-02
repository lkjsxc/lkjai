import json
import os
import shutil
import time

from benchmark_reports import validate_dense_promotion_report
from dense_debug_support import (
    MODEL_NAME,
    PACKED_CACHE_HOST,
    data_container_path,
    docker_command,
    promotion_summary,
    require_manifest_digest,
    run_json_command,
    train_args,
    train_env,
)
from run_support import ROOT, Telemetry, load_train_report, run


def promotion_paths(run_id: str, case: str, repeat: str = "repeat-01") -> tuple:
    out_dir = ROOT / "artifacts" / "benchmarks" / run_id / case / repeat
    data_dir = ROOT / "data" / "perf-runs" / run_id / case / repeat
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, data_dir


def run_train(image: str, data_dir, out_dir, steps: int, interval: float) -> tuple:
    started = time.time()
    env = train_env(data_dir, steps)
    with Telemetry(out_dir, interval):
        code = run(
            docker_command(image, "lkjai-native-train", env, train_args(steps, data_dir)),
            out_dir / "trainer.log",
            os.environ.copy(),
        )
    if code != 0:
        raise SystemExit(code)
    report = load_train_report(data_dir, out_dir / "trainer.log")
    return report, time.time() - started


def validate_artifacts(data_dir, report: dict) -> tuple:
    export_dir = data_dir / "exports" / MODEL_NAME
    checkpoint_dir = data_dir / "checkpoints" / "latest"
    require_manifest_digest(export_dir, report["export_checksum"], "export")
    require_manifest_digest(checkpoint_dir, report["checkpoint_checksum"], "checkpoint")
    if not (checkpoint_dir / "optimizer.index.json").is_file():
        raise RuntimeError(f"missing checkpoint optimizer.index.json: {checkpoint_dir}")
    return export_dir, checkpoint_dir


def run_logits_check(image: str, out_dir, export_dir, checkpoint_dir) -> dict:
    payload = run_json_command(
        image,
        "lkjai-native-logits-check",
        {},
        [
            "--model-dir",
            data_container_path(export_dir),
            "--tokens",
            "1,2,3",
            "--reference-checkpoint",
            data_container_path(checkpoint_dir),
        ],
        out_dir / "logits-reference-check.log",
    )
    if payload.get("status") != "pass":
        raise RuntimeError(f"logits check failed: {payload}")
    if payload.get("reference_check") != "pass":
        raise RuntimeError(f"reference logits check failed: {payload}")
    if float(payload.get("max_abs_diff", 1.0)) > float(payload.get("tolerance", 0.0)):
        raise RuntimeError(f"logits tolerance failed: {payload}")
    (out_dir / "logits-reference-check.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def run_resume_check(args, checkpoint_dir, expected_start: int) -> dict:
    case = f"dense_debug_resume_{args.resume_steps}"
    out_dir, data_dir = promotion_paths(args.run_id, case)
    report = run_json_command(
        args.image,
        "lkjai-native-train",
        train_env(data_dir, args.resume_steps),
        train_args(args.resume_steps, data_dir, checkpoint_dir),
        out_dir / "trainer.log",
    )
    if report.get("status") != "success":
        raise RuntimeError(f"resume run did not succeed: {report}")
    if report.get("start_step") != expected_start:
        raise RuntimeError(
            f"resume start_step mismatch: {report.get('start_step')} != {expected_start}"
        )
    shutil.copy2(data_dir / "runs" / "train-report.json", out_dir / "train-report.json")
    return report


def write_summary(run_id: str, out_dir, summary: dict) -> None:
    text = json.dumps(summary, indent=2)
    (out_dir / "promotion-summary.json").write_text(text, encoding="utf-8")
    root_out = ROOT / "artifacts" / "benchmarks" / run_id
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "promotion-summary.json").write_text(text, encoding="utf-8")


def run_promotion(args, steps: int) -> dict:
    case = f"dense_debug_train_{steps}"
    out_dir, data_dir = promotion_paths(args.run_id, case)
    if not PACKED_CACHE_HOST.is_dir():
        raise RuntimeError(f"missing packed cache: {PACKED_CACHE_HOST}")
    report, wall_elapsed = run_train(
        args.image, data_dir, out_dir, steps, args.sample_interval
    )
    validate_dense_promotion_report(report)
    shutil.copy2(data_dir / "runs" / "train-report.json", out_dir / "train-report.json")
    export_dir, checkpoint_dir = validate_artifacts(data_dir, report)
    logits_check = run_logits_check(args.image, out_dir, export_dir, checkpoint_dir)
    resume_report = run_resume_check(args, checkpoint_dir, report["optimizer_steps"])
    summary = promotion_summary(
        args.run_id, case, report, logits_check, resume_report, wall_elapsed
    )
    write_summary(args.run_id, out_dir, summary)
    return summary
