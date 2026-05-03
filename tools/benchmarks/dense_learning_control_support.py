import json
import math

from benchmark_reports import (
    dense_promotion_errors,
    is_promotable_dense_summary,
    summarize_train_report,
)
from dense_learning_control_io import (
    CASE,
    MODEL_NAME,
    app_data_path,
    build_control_cache,
    docker_command,
    run_json,
    run_train,
    write_summary,
)
from run_support import ROOT


def sampled_losses(report: dict) -> list[float]:
    values = []
    for item in report.get("loss_samples", []):
        values.append(float(item["loss"] if isinstance(item, dict) else item))
    return values


def learning_errors(
    report: dict, inspect: dict, logits: dict, infer_a: dict, infer_b: dict
) -> list[str]:
    errors = []
    samples = sampled_losses(report)
    if report.get("status") != "success":
        errors.append("train status must be success")
    if not samples or not all(math.isfinite(value) for value in samples):
        errors.append("loss_samples must be present and finite")
    initial = float(report.get("initial_loss", "nan"))
    final = float(report.get("loss", "nan"))
    if not math.isfinite(initial) or not math.isfinite(final):
        errors.append("initial_loss and loss must be finite")
    elif final > initial * 0.90:
        errors.append("final loss must be at least 10% below initial_loss")
    if float(report.get("last_quarter_loss_mean", "inf")) >= float(
        report.get("first_quarter_loss_mean", "-inf")
    ):
        errors.append("last-quarter sampled mean must be below first-quarter sampled mean")
    if report.get("weight_changed") is not True:
        errors.append("optimizer must change dense weights")
    if inspect.get("status") != "pass":
        errors.append("inspect must pass")
    if logits.get("status") != "pass" or logits.get("reference_check") != "pass":
        errors.append("logits reference check must pass")
    if infer_a.get("status") != "pass" or infer_b.get("status") != "pass":
        errors.append("dense infer must pass twice")
    if infer_a.get("checksum") != infer_b.get("checksum"):
        errors.append("dense infer checksums must match")
    return errors


def run_export_checks(image: str, out_dir, export_dir, checkpoint_dir) -> tuple:
    inspect = run_json(
        docker_command(
            image,
            "lkjai-native-inspect",
            {},
            ["--model-dir", app_data_path(export_dir)],
        ),
        out_dir / "inspect.log",
    )
    logits = run_json(
        docker_command(
            image,
            "lkjai-native-logits-check",
            {},
            [
                "--model-dir",
                app_data_path(export_dir),
                "--tokens",
                "1,2,3",
                "--reference-checkpoint",
                app_data_path(checkpoint_dir),
            ],
        ),
        out_dir / "logits-reference.log",
    )
    infer_args = ["--model-dir", app_data_path(export_dir), "--tokens", "1,2,3"]
    infer_a = run_json(
        docker_command(image, "lkjai-native-infer", {}, infer_args),
        out_dir / "infer-01.log",
    )
    infer_b = run_json(
        docker_command(image, "lkjai-native-infer", {}, infer_args),
        out_dir / "infer-02.log",
    )
    return inspect, logits, infer_a, infer_b


def write_payloads(out_dir, payloads: dict) -> None:
    for name, payload in payloads.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def summary_payload(args, report, wall_elapsed, train_command, checks) -> dict:
    inspect, logits, infer_a, infer_b = checks
    summary = summarize_train_report(report)
    learning_rejections = learning_errors(report, inspect, logits, infer_a, infer_b)
    promotion_rejections = dense_promotion_errors(report)
    summary.update(
        {
            "run_id": args.run_id,
            "case": CASE,
            "repeat": "repeat-01",
            "promotion_status": "promoted"
            if not learning_rejections and not promotion_rejections
            else "rejected",
            "returncode": 0,
            "wall_elapsed_seconds": wall_elapsed,
            "train_command": train_command,
            "docker_image": args.image,
            "learning_rejections": learning_rejections,
            "promotion_errors": promotion_rejections,
            "promotable": is_promotable_dense_summary({**summary, "returncode": 0}),
            "checksums": {
                "checkpoint": report.get("checkpoint_checksum", ""),
                "export": report.get("export_checksum", ""),
                "logits": report.get("logits_checksum", ""),
                "inspect_logits": inspect.get("logits_checksum", ""),
                "infer_01": infer_a.get("checksum", ""),
                "infer_02": infer_b.get("checksum", ""),
            },
            "logits_reference": logits,
            "infer": [infer_a, infer_b],
        }
    )
    return summary


def run_control(args) -> dict:
    out_dir = ROOT / "artifacts" / "benchmarks" / args.run_id / CASE / "repeat-01"
    data_dir = ROOT / "data" / "perf-runs" / args.run_id / CASE / "repeat-01"
    cache_dir = data_dir / "packed-cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    build_control_cache(cache_dir, args.run_id)
    report, wall_elapsed, train_command = run_train(
        args.image, data_dir, cache_dir, out_dir, args
    )
    export_dir = data_dir / "exports" / MODEL_NAME
    checkpoint_dir = data_dir / "checkpoints" / "latest"
    checks = run_export_checks(args.image, out_dir, export_dir, checkpoint_dir)
    inspect, logits, infer_a, infer_b = checks
    write_payloads(
        out_dir,
        {
            "train-report": report,
            "inspect": inspect,
            "logits-reference": logits,
            "infer-01": infer_a,
            "infer-02": infer_b,
        },
    )
    summary = summary_payload(args, report, wall_elapsed, train_command, checks)
    summary["artifact_paths"] = {
        "data_dir": str(data_dir),
        "cache": str(cache_dir),
        "checkpoint": str(checkpoint_dir),
        "export": str(export_dir),
        "logs": str(out_dir),
    }
    write_summary(out_dir, summary)
    if summary["learning_rejections"]:
        raise RuntimeError("; ".join(summary["learning_rejections"]))
    return summary
