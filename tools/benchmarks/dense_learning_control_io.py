import json
import os
import struct
import time
from pathlib import Path

from run_support import ROOT, Telemetry, load_train_report, run


CASE = "dense_learning_control_1024"
RUN_PURPOSE = "dense_learning_control"
MODEL_NAME = "dense-learning-control"
CONFIG_CONTAINER = "/workspace/configs/native/native_debug_bf16.json"
SEQ_LEN = 16
VOCAB_SIZE = 256
ROW_COUNT = 128


def app_data_path(path: Path) -> str:
    return "/app/data/" + str(path.relative_to(ROOT / "data"))


def docker_command(image: str, entrypoint: str, env: dict, args: list[str]) -> list[str]:
    command = ["docker", "run", "--rm", "--gpus", "all"]
    for key, value in sorted(env.items()):
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "-v",
            f"{ROOT / 'data'}:/app/data",
            "-v",
            f"{ROOT / 'configs'}:/workspace/configs:ro",
            "--entrypoint",
            entrypoint,
            image,
        ]
    )
    command.extend(args)
    return command


def run_json(command: list[str], log_path: Path) -> dict:
    code = run(command, log_path, os.environ.copy())
    if code != 0:
        raise SystemExit(code)
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"missing JSON payload in {log_path}")


def build_control_cache(cache_dir: Path, run_id: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens: list[int] = []
    mask: list[int] = []
    starts: list[int] = []
    for row in range(ROW_COUNT):
        starts.append(len(tokens))
        start = (row % 64) + 1
        for pos in range(SEQ_LEN):
            tokens.append(((start + pos - 1) % 64) + 1)
            mask.append(1)
    (cache_dir / "tokens.bin").write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    (cache_dir / "loss_mask.bin").write_bytes(bytes(mask))
    (cache_dir / "starts.bin").write_bytes(struct.pack(f"<{len(starts)}Q", *starts))
    metadata = {
        "format": "lkjai-packed-cache-v2",
        "run_id": run_id,
        "objective": "cyclic_bigram_control",
        "sequence_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "token_dtype": "uint16",
        "token_count": len(tokens),
        "row_count": len(starts),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    if code != 0:
        raise SystemExit(code)
    report = load_train_report(data_dir, out_dir / "train.log")
    return report, time.monotonic() - started, command


def write_summary(out_dir: Path, summary: dict) -> None:
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "learning-summary.json").write_text(text, encoding="utf-8")
    (out_dir / "benchmark-summary.json").write_text(text, encoding="utf-8")
