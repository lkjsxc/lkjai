import json
import os
from pathlib import Path

from run_support import ROOT, run


CONFIG_HOST = ROOT / "configs" / "native" / "native_debug_bf16.json"
PACKED_CACHE_HOST = (
    ROOT / "data" / "train" / "datasets" / "packed" / "train-causal_lm_full-seq1024"
)
MODEL_NAME = "dense-debug-promote"
PACKED_CACHE_CONTAINER = "/app/data/train/datasets/packed/train-causal_lm_full-seq1024"
CONFIG_CONTAINER = "/workspace/configs/native/native_debug_bf16.json"


def data_container_path(path: Path) -> str:
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
            f"{ROOT / 'corpus'}:/workspace/corpus:ro",
            "-v",
            f"{ROOT / 'configs'}:/workspace/configs:ro",
            "--entrypoint",
            entrypoint,
            image,
        ]
    )
    command.extend(args)
    return command


def run_json_command(
    image: str,
    entrypoint: str,
    env: dict,
    args: list[str],
    log_path: Path,
) -> dict:
    code = run(docker_command(image, entrypoint, env, args), log_path, os.environ.copy())
    if code != 0:
        raise SystemExit(code)
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"missing JSON payload in {log_path}")


def train_args(steps: int, data_dir: Path, resume: Path | None = None) -> list[str]:
    args = [
        "--train",
        "--mode",
        "dense",
        "--config",
        CONFIG_CONTAINER,
        "--packed-cache",
        PACKED_CACHE_CONTAINER,
        "--out",
        data_container_path(data_dir),
        "--seq-len",
        "16",
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--max-steps",
        str(steps),
        "--checkpoint-interval",
        "32",
        "--lr",
        "0.001",
    ]
    if resume is not None:
        args.extend(["--resume", data_container_path(resume)])
    return args


def train_env(data_dir: Path, steps: int) -> dict:
    return {
        "DATA_DIR": data_container_path(data_dir),
        "MODEL_NAME": MODEL_NAME,
        "TRAIN_BATCH_SIZE": "1",
        "TRAIN_COMMITTED_CORPUS_DIR": "/workspace/corpus/generated/kimi-sft-60m-v2",
        "TRAIN_GRADIENT_ACCUMULATION": "1",
        "TRAIN_LEARNING_RATE": "0.001",
        "TRAIN_MAX_OPTIMIZER_STEPS": str(steps),
        "TRAIN_NATIVE_CONFIG": CONFIG_CONTAINER,
        "TRAIN_PACKED_CACHE_DIR": PACKED_CACHE_CONTAINER,
        "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "32",
        "TRAIN_SEQUENCE_LEN": "16",
    }


def require_manifest_digest(path: Path, expected: str, kind: str) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing {kind} manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest.get("weights_checksum", "")
    if not digest:
        raise RuntimeError(f"missing {kind} weights_checksum: {manifest_path}")
    if digest != expected:
        raise RuntimeError(f"{kind} digest mismatch: {digest} != {expected}")
    for name in ["weights.lkjw", "weights.index.json"]:
        if not (path / name).is_file():
            raise RuntimeError(f"missing {kind} {name}: {path}")
    return manifest


def phase_fractions(report: dict) -> dict:
    timings = report.get("timings", {})
    elapsed = float(report.get("elapsed_seconds", 0.0))
    if elapsed <= 0.0:
        return {key: 0.0 for key in timings}
    return {key: float(value) / elapsed for key, value in timings.items()}


def promotion_summary(
    run_id: str,
    case: str,
    report: dict,
    logits_check: dict,
    resume_report: dict,
    elapsed_seconds: float,
) -> dict:
    config = json.loads(CONFIG_HOST.read_text(encoding="utf-8"))
    timings = report.get("timings", {})
    elapsed = float(report.get("elapsed_seconds", 0.0))
    return {
        "run_id": run_id,
        "case": case,
        "promotion_status": "promoted",
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "model_kind": report.get("model_kind"),
        "accepted_cuda_training": report.get("accepted_cuda_training"),
        "implementation_status": report.get("implementation_status"),
        "device": report.get("cuda_device_name", ""),
        "backend": {
            "forward": report.get("forward_backend", ""),
            "backward": report.get("backward_backend", ""),
            "optimizer": report.get("optimizer_backend", ""),
        },
        "batch_size": report.get("batch_size"),
        "seq_len": report.get("seq_len"),
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "parameter_count": report.get("parameter_count"),
        "optimizer_steps": report.get("optimizer_steps"),
        "initial_loss": report.get("initial_loss"),
        "loss": report.get("loss"),
        "tokens_per_second": report.get("tokens_per_second"),
        "elapsed_seconds": elapsed,
        "wall_elapsed_seconds": elapsed_seconds,
        "h2d_fraction": float(timings.get("h2d", 0.0)) / elapsed
        if elapsed > 0.0
        else 0.0,
        "phase_fractions": phase_fractions(report),
        "checksums": {
            "checkpoint": report.get("checkpoint_checksum", ""),
            "export": report.get("export_checksum", ""),
            "logits": report.get("logits_checksum", ""),
        },
        "logits_check": logits_check,
        "resume_check": {
            "status": resume_report.get("status"),
            "start_step": resume_report.get("start_step"),
            "optimizer_steps": resume_report.get("optimizer_steps"),
        },
        "artifact_paths": {
            "checkpoint": report.get("checkpoint_path", ""),
            "export": report.get("export_path", ""),
            "train_report": "train-report.json",
        },
    }
