import json
import os
import time
from pathlib import Path

from dense_2h_paths import (
    app_data_path,
    config_container_path,
    docker_command,
    host_path,
    run_json,
)
from run_support import ROOT, Telemetry, load_train_report, run


CASE = "dense_2h_bf16_cuda"
RUN_PURPOSE = "accepted_training"


def sample_interval(args, steps: int) -> int:
    if args.loss_sample_interval > 0:
        return args.loss_sample_interval
    return max(1, steps // 16)


def train_args(args, data_dir: Path, steps: int) -> list[str]:
    checkpoint_interval = min(args.checkpoint_interval, steps)
    return [
        "--train",
        "--mode",
        "dense",
        "--run-purpose",
        RUN_PURPOSE,
        "--config",
        config_container_path(host_path(args.native_config)),
        "--packed-cache",
        app_data_path(host_path(args.cache)),
        "--out",
        app_data_path(data_dir),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-steps",
        str(steps),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--loss-sample-interval",
        str(sample_interval(args, steps)),
        "--lr",
        str(args.lr),
    ]


def train_env(args, data_dir: Path, steps: int) -> dict:
    return {
        "DATA_DIR": app_data_path(data_dir),
        "MODEL_NAME": args.model_name,
        "TRAIN_BATCH_SIZE": str(args.batch_size),
        "TRAIN_GRADIENT_ACCUMULATION": str(args.grad_accum),
        "TRAIN_LEARNING_RATE": str(args.lr),
        "TRAIN_LOSS_SAMPLE_INTERVAL": str(sample_interval(args, steps)),
        "TRAIN_MAX_OPTIMIZER_STEPS": str(steps),
        "TRAIN_NATIVE_CONFIG": config_container_path(host_path(args.native_config)),
        "TRAIN_PACKED_CACHE_DIR": app_data_path(host_path(args.cache)),
        "TRAIN_RUN_PURPOSE": RUN_PURPOSE,
        "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": str(
            min(args.checkpoint_interval, steps)
        ),
        "TRAIN_SEQUENCE_LEN": str(args.seq_len),
        "TRAIN_SEED": str(args.seed),
    }


def failed_report(args, data_dir: Path) -> dict:
    return {
        "schema_version": 3,
        "trainer_mode": "train",
        "run_purpose": RUN_PURPOSE,
        "status": "fail",
        "model_kind": "dense",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "config_path": config_container_path(host_path(args.native_config)),
        "packed_cache_path": app_data_path(host_path(args.cache)),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "grad_accum": args.grad_accum,
        "optimizer_steps": 0,
        "microsteps": 0,
        "tokens_seen": 0,
        "loss_tokens": 0,
        "initial_loss": 0.0,
        "loss": 0.0,
        "loss_samples": [],
        "elapsed_seconds": 0.0,
        "tokens_per_second": 0.0,
        "checkpoint_path": app_data_path(data_dir / "checkpoints" / "latest"),
        "export_path": app_data_path(data_dir / "exports" / args.model_name),
        "timings": {},
    }


def run_train_phase(args, phase: str, steps: int, out_dir: Path):
    data_dir = ROOT / "data" / "perf-runs" / args.run_id / CASE / phase
    data_dir.mkdir(parents=True, exist_ok=True)
    env = train_env(args, data_dir, steps)
    cli_args = train_args(args, data_dir, steps)
    command = docker_command(args.image, "lkjai-native-train", env, cli_args)
    (out_dir / f"{phase}-train-command.json").write_text(
        json.dumps({"command": command, "env": env, "args": cli_args}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    with Telemetry(out_dir / phase, args.sample_interval):
        code = run(command, out_dir / f"{phase}-train.log", os.environ.copy())
    wall_elapsed = time.monotonic() - started
    try:
        report = load_train_report(data_dir, out_dir / f"{phase}-train.log")
    except FileNotFoundError:
        report = failed_report(args, data_dir)
    (out_dir / f"{phase}-train-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report, wall_elapsed, code, command, data_dir


def export_checks(args, phase: str, out_dir: Path, data_dir: Path) -> dict:
    export_dir = data_dir / "exports" / args.model_name
    checkpoint_dir = data_dir / "checkpoints" / "latest"
    inspect = run_json(
        docker_command(
            args.image,
            "lkjai-native-inspect",
            {},
            ["--model-dir", app_data_path(export_dir)],
        ),
        out_dir / f"{phase}-inspect.log",
    )
    logits = run_json(
        docker_command(
            args.image,
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
        out_dir / f"{phase}-logits-reference.log",
    )
    infer_args = ["--model-dir", app_data_path(export_dir), "--tokens", "1,2,3"]
    infer_a = run_json(
        docker_command(args.image, "lkjai-native-infer", {}, infer_args),
        out_dir / f"{phase}-infer-01.log",
    )
    infer_b = run_json(
        docker_command(args.image, "lkjai-native-infer", {}, infer_args),
        out_dir / f"{phase}-infer-02.log",
    )
    checks = {"inspect": inspect, "logits_reference": logits, "infer": [infer_a, infer_b]}
    (out_dir / f"{phase}-export-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    return checks
