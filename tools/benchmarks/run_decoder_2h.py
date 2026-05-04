#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from dense_2h_paths import (
    app_data_path,
    build_and_validate_cache,
    build_image,
    config_container_path,
    docker_command,
    host_path,
    run_json,
)
from run_support import ROOT, Telemetry, load_train_report, run


CASE = "decoder_2h_bf16_cuda"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default="data/train/datasets/train.jsonl")
    parser.add_argument("--tokenizer", default="data/train/tokenizer/tokenizer.json")
    parser.add_argument("--native-config", default="configs/native/decoder_18m_bf16_3070.json")
    parser.add_argument("--cache", default="data/train/datasets/packed/train-causal_lm_full-seq1024")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sequence-count", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--target-seconds", type=int, default=7200)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--checkpoint-interval", type=int, default=512)
    parser.add_argument("--out-dir", default="artifacts/benchmarks")
    parser.add_argument("--image", default="lkjai-native-decoder-bench")
    parser.add_argument("--cuda-arch-flags", default="86-real;86-virtual")
    parser.add_argument("--model-name", default="decoder-2h-18m-3070")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--skip-cache-build", action="store_true")
    return parser.parse_args()


def train_command(args, data_dir: Path) -> list[str]:
    env = {
        "DATA_DIR": app_data_path(data_dir),
        "MODEL_NAME": args.model_name,
        "TRAIN_MODEL_KIND": "decoder",
        "TRAIN_TARGET_SECONDS": str(args.target_seconds),
    }
    cli = [
        "--train", "--mode", "decoder", "--run-purpose", "decoder_2h",
        "--config", config_container_path(host_path(args.native_config)),
        "--packed-cache", app_data_path(host_path(args.cache)),
        "--out", app_data_path(data_dir), "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
        "--max-steps", str(args.max_steps), "--target-seconds", str(args.target_seconds),
        "--checkpoint-interval", str(args.checkpoint_interval), "--lr", str(args.lr),
    ]
    return docker_command(args.image, "lkjai-native-train", env, cli)


def export_checks(args, out_dir: Path, data_dir: Path) -> dict:
    export_dir = data_dir / "exports" / args.model_name
    inspect = run_json(docker_command(args.image, "lkjai-native-inspect", {},
        ["--model-dir", app_data_path(export_dir)]), out_dir / "inspect.log")
    logits = run_json(docker_command(args.image, "lkjai-native-logits-check", {},
        ["--model-dir", app_data_path(export_dir), "--tokens", "1,2,3"]),
        out_dir / "logits.log")
    return {"inspect": inspect, "logits": logits}


def run_full(args, out_dir: Path) -> dict:
    data_dir = ROOT / "data" / "perf-runs" / args.run_id / CASE / "full"
    data_dir.mkdir(parents=True, exist_ok=True)
    command = train_command(args, data_dir)
    (out_dir / "train-command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with Telemetry(out_dir, args.sample_interval):
        code = run(command, out_dir / "train.log", os.environ.copy())
    wall = time.monotonic() - started
    try:
        report = load_train_report(data_dir, out_dir / "train.log")
    except FileNotFoundError:
        report = {"status": "fail", "model_kind": "decoder"}
    checks = export_checks(args, out_dir, data_dir) if code == 0 else {}
    return {"returncode": code, "wall_elapsed_seconds": wall,
            "report": report, "checks": checks, "data_dir": str(data_dir)}


def main():
    args = parse_args()
    out_dir = host_path(args.out_dir) / args.run_id / CASE / "repeat-01"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        build_image(args.image, args.cuda_arch_flags, out_dir)
    cache = build_and_validate_cache(args, out_dir)
    summary = {"report_kind": CASE, "run_id": args.run_id,
               "target_seconds": args.target_seconds, "cache": cache,
               "full_status": "not_requested"}
    if args.full:
        full = run_full(args, out_dir)
        summary["full"] = full
        summary["full_status"] = "completed" if full["returncode"] == 0 else "failed"
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "decoder-2h-summary.json").write_text(text, encoding="utf-8")
    (out_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")
    print(json.dumps({"runner_status": summary["full_status"], "summary": str(out_dir)}))


if __name__ == "__main__":
    main()
