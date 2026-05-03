#!/usr/bin/env python3
import argparse
import json

from dense_2h_paths import build_and_validate_cache, build_image, host_path
from dense_2h_report import (
    phase_summary,
    target_steps_from_pilot,
    train_blocker,
    write_markdown,
)
from dense_2h_train import CASE, export_checks, run_train_phase
from dense_accepted_training_cache import cache_summary


REPEAT = "repeat-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default="data/train/datasets/train.jsonl")
    parser.add_argument("--tokenizer", default="data/train/tokenizer/tokenizer.json")
    parser.add_argument(
        "--native-config",
        default="configs/native/native_dense_20m_bf16_3070.json",
    )
    parser.add_argument(
        "--cache",
        default="data/train/datasets/packed/train-causal_lm_full-seq1024",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sequence-count", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--pilot-steps", type=int, default=128)
    parser.add_argument("--target-seconds", type=int, default=7200)
    parser.add_argument("--out-dir", default="artifacts/benchmarks")
    parser.add_argument("--image", default="lkjai-native-bench")
    parser.add_argument("--cuda-arch-flags", default="86-real;86-virtual")
    parser.add_argument("--model-name", default="dense-2h-20m-3070")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--checkpoint-interval", type=int, default=512)
    parser.add_argument("--loss-sample-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--skip-cache-build", action="store_true")
    return parser.parse_args()


def summarize_pilot(args, out_dir, cache_metadata):
    pilot_report, pilot_wall, pilot_code, pilot_command, pilot_data = run_train_phase(
        args, "pilot", args.pilot_steps, out_dir
    )
    pilot_checks = (
        export_checks(args, "pilot", out_dir, pilot_data)
        if pilot_report.get("status") == "success"
        else {}
    )
    target_steps = (
        target_steps_from_pilot(pilot_report, args.target_seconds)
        if pilot_report.get("status") == "success"
        else 0
    )
    summary = {
        "report_kind": "dense_2h_bf16_cuda",
        "run_id": args.run_id,
        "case": CASE,
        "repeat": REPEAT,
        "runner_status": (
            "pass"
            if pilot_code == 0 and pilot_report.get("status") == "success"
            else "blocked"
        ),
        "full_status": "not_requested",
        "target_seconds": args.target_seconds,
        "target_optimizer_steps": target_steps,
        "docker_image": args.image,
        "cache": cache_summary(cache_metadata),
        "pilot_command": pilot_command,
        "pilot": phase_summary(
            pilot_report, pilot_wall, pilot_code, pilot_data, args.model_name
        ),
        "pilot_checks": pilot_checks,
        "limitations": pilot_report.get("limitations", []),
        "blocker": "",
    }
    if pilot_code != 0 or pilot_report.get("status") != "success":
        summary["blocker"] = train_blocker("pilot", pilot_code, pilot_report)
    return summary


def run_full_if_requested(args, out_dir, summary):
    if summary["blocker"] or not args.full:
        summary["full_status"] = (
            summary["full_status"] if summary["blocker"] else "skipped_without_full_flag"
        )
        return
    full_report, full_wall, full_code, full_command, full_data = run_train_phase(
        args, "full", summary["target_optimizer_steps"], out_dir
    )
    full_checks = (
        export_checks(args, "full", out_dir, full_data)
        if full_report.get("status") == "success"
        else {}
    )
    full_ok = full_code == 0 and full_report.get("status") == "success"
    summary.update(
        {
            "runner_status": "pass" if full_ok else "blocked",
            "full_status": "completed" if full_ok else "failed",
            "full_command": full_command,
            "full": phase_summary(
                full_report, full_wall, full_code, full_data, args.model_name
            ),
            "full_checks": full_checks,
            "limitations": full_report.get("limitations", []),
        }
    )
    if not full_ok:
        summary["blocker"] = train_blocker("full", full_code, full_report)


def main() -> None:
    args = parse_args()
    if args.pilot_steps <= 0 or args.target_seconds <= 0:
        raise SystemExit("--pilot-steps and --target-seconds must be positive")
    out_dir = host_path(args.out_dir) / args.run_id / CASE / REPEAT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot").mkdir(parents=True, exist_ok=True)
    (out_dir / "full").mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        build_image(args.image, args.cuda_arch_flags, out_dir)
    cache_metadata = build_and_validate_cache(args, out_dir)
    summary = summarize_pilot(args, out_dir, cache_metadata)
    run_full_if_requested(args, out_dir, summary)
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "dense-2h-summary.json").write_text(text, encoding="utf-8")
    (out_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")
    write_markdown(summary, out_dir / "dense-2h-report.md")
    print(json.dumps({"runner_status": summary["runner_status"], "summary": str(out_dir)}))


if __name__ == "__main__":
    main()
