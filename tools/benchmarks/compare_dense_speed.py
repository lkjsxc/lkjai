#!/usr/bin/env python3
import json
import sys
from pathlib import Path


MATCH_FIELDS = (
    "config_digest",
    "dataset_digest",
    "batch_size",
    "seq_len",
    "grad_accum",
    "cuda_arch_flags",
)


def load_report(path: str) -> dict:
    target = Path(path)
    if target.is_dir():
        target = target / "runs" / "train-report.json"
    return json.loads(target.read_text(encoding="utf-8"))


def value(report: dict, key: str):
    if key in report:
        return report[key]
    if key == "config_digest":
        return report.get("config", {}).get("digest", "")
    if key == "dataset_digest":
        return report.get("packed_cache_digest", report.get("cache_digest", ""))
    return ""


def errors(before: dict, after: dict) -> list[str]:
    found = []
    for key in MATCH_FIELDS:
        if value(before, key) != value(after, key):
            found.append(f"{key} differs")
    if after.get("accepted_cuda_training") is not True:
        found.append("post report is not accepted dense CUDA training")
    if after.get("model_kind", "dense") != "dense":
        found.append("post report model_kind is not dense")
    return found


def metric(report: dict, key: str) -> float:
    if key == "backward":
        return float(report.get("timings", {}).get("backward", 0.0))
    return float(report.get(key, 0.0))


def compare(before: dict, after: dict) -> dict:
    before_backward = metric(before, "backward")
    after_backward = metric(after, "backward")
    before_tps = metric(before, "tokens_per_second")
    after_tps = metric(after, "tokens_per_second")
    backward_speedup = (
        before_backward / after_backward if after_backward > 0.0 else 0.0
    )
    throughput_ratio = after_tps / before_tps if before_tps > 0.0 else 0.0
    mismatch = errors(before, after)
    accepted = not mismatch and (
        after_backward < before_backward or throughput_ratio >= 0.95
    )
    return {
        "accepted": accepted,
        "errors": mismatch,
        "before_backward_seconds": before_backward,
        "after_backward_seconds": after_backward,
        "backward_speedup": backward_speedup,
        "before_tokens_per_second": before_tps,
        "after_tokens_per_second": after_tps,
        "throughput_ratio": throughput_ratio,
        "post_backward_backend": after.get("backward_backend", ""),
        "post_embedding_grad_backend": after.get("embedding_grad_backend", ""),
        "post_loss_kernel_backend": after.get("loss_kernel_backend", ""),
        "post_batch_staging_backend": after.get("batch_staging_backend", ""),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_dense_speed.py BASELINE_REPORT POST_REPORT", file=sys.stderr)
        return 2
    result = compare(load_report(sys.argv[1]), load_report(sys.argv[2]))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
