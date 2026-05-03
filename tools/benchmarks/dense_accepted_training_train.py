import json
import os
import time
from pathlib import Path

from dense_accepted_training_io import (
    CONFIG_CONTAINER,
    MODEL_NAME,
    RUN_PURPOSE,
    SEQ_LEN,
    app_data_path,
    docker_command,
    run_json,
)
from run_support import Telemetry, load_train_report, run


def train_env(data_dir: Path, cache_dir: Path, args) -> dict:
    return {
        "DATA_DIR": app_data_path(data_dir),
        "MODEL_NAME": MODEL_NAME,
        "TRAIN_BATCH_SIZE": "4",
        "TRAIN_GRADIENT_ACCUMULATION": "1",
        "TRAIN_LEARNING_RATE": str(args.lr),
        "TRAIN_LOSS_SAMPLE_INTERVAL": str(args.loss_sample_interval),
        "TRAIN_MAX_OPTIMIZER_STEPS": str(args.steps),
        "TRAIN_NATIVE_CONFIG": CONFIG_CONTAINER,
        "TRAIN_PACKED_CACHE_DIR": app_data_path(cache_dir),
        "TRAIN_RUN_PURPOSE": RUN_PURPOSE,
        "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "128",
        "TRAIN_SEQUENCE_LEN": str(SEQ_LEN),
    }


def train_args(data_dir: Path, cache_dir: Path, args) -> list[str]:
    return [
        "--train",
        "--mode",
        "dense",
        "--run-purpose",
        RUN_PURPOSE,
        "--config",
        CONFIG_CONTAINER,
        "--packed-cache",
        app_data_path(cache_dir),
        "--out",
        app_data_path(data_dir),
        "--seq-len",
        str(SEQ_LEN),
        "--batch-size",
        "4",
        "--grad-accum",
        "1",
        "--max-steps",
        str(args.steps),
        "--checkpoint-interval",
        "128",
        "--loss-sample-interval",
        str(args.loss_sample_interval),
        "--lr",
        str(args.lr),
    ]


def minimal_failed_report(args, data_dir: Path, cache_dir: Path) -> dict:
    return {
        "schema_version": 3,
        "trainer_mode": "train",
        "run_purpose": RUN_PURPOSE,
        "status": "fail",
        "model_kind": "dense",
        "accepted_cuda_training": True,
        "implementation_status": "accepted",
        "config_path": CONFIG_CONTAINER,
        "packed_cache_path": app_data_path(cache_dir),
        "batch_size": 4,
        "seq_len": SEQ_LEN,
        "grad_accum": 1,
        "optimizer_steps": 0,
        "microsteps": 0,
        "tokens_seen": 0,
        "loss_tokens": 0,
        "initial_loss": 0.0,
        "loss": 0.0,
        "loss_samples": [],
        "loss_sample_interval": args.loss_sample_interval,
        "learning_status": "unknown",
        "elapsed_seconds": 0.0,
        "tokens_per_second": 0.0,
        "checkpoint_path": app_data_path(data_dir / "checkpoints" / "latest"),
        "export_path": app_data_path(data_dir / "exports" / MODEL_NAME),
        "timings": {},
    }


def run_train(image: str, data_dir: Path, cache_dir: Path, out_dir: Path, args):
    env = train_env(data_dir, cache_dir, args)
    cli_args = train_args(data_dir, cache_dir, args)
    command = docker_command(image, "lkjai-native-train", env, cli_args)
    payload = {
        "image": image,
        "entrypoint": "lkjai-native-train",
        "command": command,
        "env": env,
        "args": cli_args,
    }
    (out_dir / "train-command.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    with Telemetry(out_dir, args.sample_interval):
        code = run(command, out_dir / "train.log", os.environ.copy())
    try:
        report = load_train_report(data_dir, out_dir / "train.log")
    except FileNotFoundError:
        report = minimal_failed_report(args, data_dir, cache_dir)
    return report, time.monotonic() - started, command, code


def run_export_checks(image: str, out_dir: Path, export_dir: Path, checkpoint_dir: Path):
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
