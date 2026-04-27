from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path


DEFAULT_RUN_DIR = Path("runs/kimi_corpus")
DEFAULT_OUTPUT_DIR = Path("data/kimi_synthetic")


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    config: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = parse_scalar(value.strip())
    return config


def parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [parse_scalar(item.strip()) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value.strip("\"'")


def apply_overrides(config: dict, args: Namespace) -> dict:
    for key in [
        "target_tokens",
        "mode",
        "pretrain_ratio",
        "sft_ratio",
        "output_dir",
        "prompt_version",
        "timeout_seconds",
        "sleep_between_calls",
        "max_retries",
        "max_calls",
        "batch_documents",
        "sample_documents",
        "stop_file",
        "parallelism",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if getattr(args, "quarantine_bad_shards", False):
        config["quarantine_bad_shards"] = True
    defaults = {
        "target_tokens": 60_000_000,
        "mode": "sft",
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "prompt_version": "v1",
        "pretrain_ratio": 0.0,
        "sft_ratio": 1.0,
        "batch_documents": 12,
        "sample_documents": 20,
        "timeout_seconds": 240,
        "sleep_between_calls": 1.0,
        "max_retries": 2,
        "repair_attempts": 1,
        "parallelism": 1,
        "quarantine_bad_shards": True,
        "stop_file": str(DEFAULT_RUN_DIR / "STOP"),
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config
