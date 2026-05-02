#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import statistics
import time

from run_support import (
    ROOT,
    Telemetry,
    build_image,
    load_train_report,
    prepare_data_dir,
    is_promotable_dense_summary,
    run,
    summarize_train_report,
)


CASES = {
    "dense_smoke_2": {
        "mode": "smoke",
        "env": {"TRAIN_MAX_OPTIMIZER_STEPS": "2"},
    },
    "dense_packed_train_2": {
        "mode": "train",
        "env": {
            "TRAIN_MAX_OPTIMIZER_STEPS": "2",
            "TRAIN_SEQUENCE_LEN": "1024",
        },
    },
}



def run_case(image: str, run_id: str, case: str, repeat: int, base_env: dict, sample_interval: float) -> dict:
    out_dir = ROOT / "artifacts" / "benchmarks" / run_id / case / f"repeat-{repeat:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "perf-runs" / run_id / case / f"repeat-{repeat:02d}"
    prepare_data_dir(data_dir)
    env = os.environ.copy()
    env.update(base_env)
    env.update(CASES[case]["env"])
    env.update(
        {
            "DATA_DIR": "/app/data/perf-runs/" + f"{run_id}/{case}/repeat-{repeat:02d}",
            "TRAIN_COMMITTED_CORPUS_DIR": "/workspace/corpus/generated/kimi-sft-60m-v2",
        }
    )
    steps = env.get("TRAIN_MAX_OPTIMIZER_STEPS", "2")
    train_args = (
        ["--smoke", "--steps", steps]
        if CASES[case]["mode"] == "smoke"
        else [
            "--train",
            "--config",
            env["TRAIN_NATIVE_CONFIG"],
            "--seq-len",
            env.get("TRAIN_SEQUENCE_LEN", "1024"),
            "--max-steps",
            steps,
        ]
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--entrypoint",
        "lkjai-native-train",
        "-v",
        f"{ROOT / 'data'}:/app/data",
        "-v",
        f"{ROOT / 'corpus'}:/workspace/corpus:ro",
        "-v",
        f"{ROOT / 'configs'}:/workspace/configs:ro",
        image,
    ] + train_args
    docker_env = []
    for key, value in sorted(env.items()):
        if key.startswith("TRAIN_") or key in {"DATA_DIR", "MODEL_NAME"}:
            docker_env.extend(["-e", f"{key}={value}"])
    command = command[:5] + docker_env + command[5:]
    started = time.time()
    with Telemetry(out_dir, sample_interval):
        code = run(command, out_dir / "trainer.log", env=os.environ.copy())
    elapsed = time.time() - started
    summary = {
        "run_id": run_id,
        "case": case,
        "repeat": repeat,
        "returncode": code,
        "elapsed_seconds": elapsed,
        "env": {key: env[key] for key in sorted(env) if key.startswith("TRAIN_")},
        "data_dir": str(data_dir),
    }
    report = load_train_report(data_dir, out_dir / "trainer.log")
    report_path = data_dir / "runs" / "train-report.json"
    if report_path.exists():
        shutil.copy2(report_path, out_dir / "train-report.json")
    else:
        (out_dir / "train-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    summary.update(summarize_train_report(report))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_summary(run_id: str, rows: list[dict]) -> None:
    out_dir = ROOT / "artifacts" / "benchmarks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    aggregate = []
    for case, case_rows in grouped.items():
        accepted_rows = [row for row in case_rows if is_promotable_dense_summary(row)]
        medians = [row.get("median_step_seconds", 0.0) for row in accepted_rows]
        toks = [row.get("median_tokens_per_second", 0.0) for row in accepted_rows]
        aggregate.append(
            {
                "case": case,
                "runs": len(case_rows),
                "successful_runs": sum(1 for row in case_rows if row.get("returncode") == 0),
                "accepted_cuda_runs": len(accepted_rows),
                "median_step_seconds": statistics.median(medians) if medians else 0.0,
                "median_tokens_per_second": statistics.median(toks) if toks else 0.0,
            }
        )
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--image", default="")
    parser.add_argument("--cases", default="dense_smoke_2,dense_packed_train_2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--warmup-microsteps", type=int, default=2)
    args = parser.parse_args()

    image = args.image or f"lkjai-train-perf:{args.run_id}"
    if not args.no_build:
        build_image(image)
    base_env = {
        "MODEL_NAME": "lkjai-scratch-40m",
        "TRAIN_PRESET": "agent",
        "TRAIN_CONFIG": "/workspace/configs/training/scratch_40m_12h.json",
        "TRAIN_NATIVE_CONFIG": "/workspace/configs/native/native_debug_bf16.json",
        "TRAIN_MAX_OPTIMIZER_STEPS": str(args.max_optimizer_steps),
        "TRAIN_GRADIENT_ACCUMULATION": str(args.gradient_accumulation),
    }
    rows = []
    for case in [item.strip() for item in args.cases.split(",") if item.strip()]:
        if case not in CASES:
            raise SystemExit(f"unknown case {case}")
        for repeat in range(1, args.repeats + 1):
            rows.append(run_case(image, args.run_id, case, repeat, base_env, args.sample_interval))
            write_summary(args.run_id, rows)
    print(json.dumps({"run_id": args.run_id, "rows": len(rows), "status": "pass"}))


if __name__ == "__main__":
    main()
