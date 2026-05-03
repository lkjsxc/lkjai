import json
import math
from pathlib import Path

from accepted_training_reports import accepted_training_summary_errors
from benchmark_reports import (
    dense_promotion_errors,
    is_promotable_dense_summary,
    summarize_train_report,
)
from dense_accepted_training_cache import build_and_validate_cache, cache_summary
from dense_accepted_training_io import (
    CASE,
    CONFIG_HOST,
    MODEL_NAME,
    SEQ_LEN,
    SEQUENCE_COUNT,
    SOURCE_HOST,
    TOKENIZER_HOST,
    write_payloads,
)
from dense_accepted_training_train import run_export_checks, run_train
from run_support import ROOT


def token_accounting(report: dict) -> dict:
    tokens_seen = int(report.get("tokens_seen", report.get("input_tokens", 0)))
    loss_tokens = int(report.get("loss_tokens", 0))
    return {
        "tokens_seen": tokens_seen,
        "loss_tokens": loss_tokens,
        "non_loss_tokens": max(0, tokens_seen - loss_tokens),
        "loss_token_fraction": loss_tokens / tokens_seen if tokens_seen > 0 else 0.0,
        "valid": tokens_seen > 0 and 0 < loss_tokens < tokens_seen,
    }


def sampled_losses(report: dict) -> list[float]:
    values = []
    for item in report.get("loss_samples", []):
        values.append(float(item["loss"] if isinstance(item, dict) else item))
    return values


def lr_selection_errors(report: dict, inspect: dict, logits: dict, infer_a: dict, infer_b: dict) -> list[str]:
    errors = []
    samples = sampled_losses(report)
    if report.get("status") != "success":
        errors.append("train status must be success")
    if not samples or not all(math.isfinite(value) for value in samples):
        errors.append("loss samples must be present and finite")
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
    if inspect.get("status") != "pass":
        errors.append("inspect must pass")
    if logits.get("status") != "pass" or logits.get("reference_check") != "pass":
        errors.append("BF16 export/reference logits check must pass")
    if infer_a.get("status") != "pass" or infer_b.get("status") != "pass":
        errors.append("dense infer must pass twice")
    if infer_a.get("checksum") != infer_b.get("checksum"):
        errors.append("dense infer checksums must match")
    return errors


def validation_report(report: dict, cache: dict, infer: list[dict]) -> dict:
    enriched = dict(report)
    enriched["cache_row_count"] = cache.get("row_count", 0)
    enriched["source_digest"] = cache.get("source_digest", "")
    enriched["tokenizer_digest"] = cache.get("tokenizer_digest", "")
    enriched["config_digest"] = cache.get("config_digest", "")
    enriched["dataset_digest"] = cache.get("packed_data_checksum", "")
    enriched["infer"] = infer
    return enriched


def summary_payload(args, report: dict, wall_elapsed: float, train_command: list[str], cache_metadata: dict, checks: tuple, returncode: int) -> dict:
    inspect, logits, infer_a, infer_b = checks
    cache = cache_summary(cache_metadata)
    summary = summarize_train_report(report)
    summary.update(
        {
            "report_kind": "accepted_training",
            "run_id": args.run_id,
            "case": CASE,
            "repeat": "repeat-01",
            "model_name": MODEL_NAME,
            "learning_rate": float(args.lr),
            "returncode": returncode,
            "wall_elapsed_seconds": wall_elapsed,
            "train_command": train_command,
            "docker_image": args.image,
            "cache_metadata": cache,
            "token_accounting": token_accounting(report),
            "checksums": {
                "checkpoint": report.get("checkpoint_checksum", ""),
                "export": report.get("export_checksum", ""),
                "logits": report.get("logits_checksum", ""),
                "inspect_logits": inspect.get("logits_checksum", ""),
                "infer_01": infer_a.get("checksum", ""),
                "infer_02": infer_b.get("checksum", ""),
                "packed_cache": cache.get("packed_data_checksum", ""),
            },
            "inspect": inspect,
            "logits_reference": logits,
            "infer": [infer_a, infer_b],
        }
    )
    lr_errors = lr_selection_errors(report, inspect, logits, infer_a, infer_b)
    foundation_errors = dense_promotion_errors(
        validation_report(report, cache, [infer_a, infer_b])
    )
    accepted_errors = accepted_training_summary_errors(summary)
    rejection_reasons = sorted(set(lr_errors + foundation_errors + accepted_errors))
    summary.update(
        {
            "lr_selection_status": "pass" if not lr_errors else "fail",
            "lr_selection_errors": lr_errors,
            "promotion_errors": foundation_errors,
            "accepted_training_errors": accepted_errors,
            "rejection_reasons": rejection_reasons,
            "promotion_status": "promoted" if not rejection_reasons else "rejected",
        }
    )
    summary["promotable"] = is_promotable_dense_summary(summary)
    if not summary["promotable"] and summary["promotion_status"] == "promoted":
        summary["promotion_status"] = "rejected"
        summary["rejection_reasons"].append("promotable summary validation returned false")
    return summary


def write_summary(out_dir: Path, summary: dict, data_dir: Path | None = None) -> None:
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "accepted-training-summary.json").write_text(text, encoding="utf-8")
    (out_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")
    if data_dir is not None:
        (data_dir / "accepted-training-summary.json").write_text(text, encoding="utf-8")
        (data_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")


def run_accepted(args) -> dict:
    for path, label in ((SOURCE_HOST, "source JSONL"), (TOKENIZER_HOST, "tokenizer"), (CONFIG_HOST, "native config")):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    out_dir = ROOT / "artifacts" / "benchmarks" / args.run_id / CASE / "repeat-01"
    data_dir = ROOT / "data" / "perf-runs" / args.run_id / CASE / "repeat-01"
    cache_dir = data_dir / "packed-cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_metadata = build_and_validate_cache(
        args.run_id, args.sequence_count, cache_dir, out_dir
    )
    report, wall_elapsed, train_command, returncode = run_train(
        args.image, data_dir, cache_dir, out_dir, args
    )
    export_dir = data_dir / "exports" / MODEL_NAME
    checkpoint_dir = data_dir / "checkpoints" / "latest"
    if returncode == 0 and report.get("status") == "success":
        checks = run_export_checks(args.image, out_dir, export_dir, checkpoint_dir)
    else:
        skipped = {"status": "skipped", "reason": "training failed"}
        checks = (skipped, skipped, skipped, skipped)
    inspect, logits, infer_a, infer_b = checks
    write_payloads(out_dir, {"train-report": report, "inspect": inspect, "logits-reference": logits, "infer-01": infer_a, "infer-02": infer_b})
    summary = summary_payload(
        args, report, wall_elapsed, train_command, cache_metadata, checks, returncode
    )
    summary["artifact_paths"] = {"data_dir": str(data_dir), "cache": str(cache_dir), "checkpoint": str(checkpoint_dir), "export": str(export_dir), "logs": str(out_dir)}
    write_summary(out_dir, summary, data_dir)
    if returncode != 0:
        raise SystemExit(returncode)
    return summary
