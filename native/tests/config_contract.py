#!/usr/bin/env python3
import json
import sys
from pathlib import Path


NATIVE_KEYS = {
    "model",
    "model_kind",
    "dtype",
    "vocab_size",
    "context",
    "layers",
    "hidden_size",
    "heads",
    "kv_heads",
    "head_dim",
    "ffn_size",
    "activation",
    "rope_theta",
    "rms_norm_eps",
    "tie_embeddings",
    "seed",
}
TRAIN_KEYS = {
    "format",
    "name",
    "description",
    "preset",
    "model_name",
    "model_kind",
    "native_config",
    "packed_cache_dir",
    "objective",
    "sequence_len",
    "learning_rate",
    "warmup_steps",
    "batch_size",
    "gradient_accumulation",
    "max_optimizer_steps",
    "save_latest_every_optimizer_steps",
    "target_seconds",
    "seed",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_native(path: Path) -> dict:
    cfg = load(path)
    extra = sorted(set(cfg) - NATIVE_KEYS)
    require(not extra, f"{path}: unknown keys {extra}")
    required = NATIVE_KEYS - {"model_kind"}
    missing = sorted(required - set(cfg))
    require(not missing, f"{path}: missing keys {missing}")
    require(cfg["dtype"] == "bf16", f"{path}: dtype must be bf16")
    require(cfg["activation"] == "swiglu", f"{path}: activation must be swiglu")
    for key in [
        "vocab_size",
        "context",
        "layers",
        "hidden_size",
        "heads",
        "kv_heads",
        "head_dim",
        "ffn_size",
    ]:
        require(isinstance(cfg[key], int) and cfg[key] > 0, f"{path}: bad {key}")
    require(1 < cfg["context"] <= 4096, f"{path}: context out of bounds")
    require(0 < cfg["vocab_size"] <= 16384, f"{path}: vocab out of bounds")
    require(
        cfg["heads"] * cfg["head_dim"] == cfg["hidden_size"],
        f"{path}: heads * head_dim must equal hidden_size",
    )
    require(
        cfg["heads"] % cfg["kv_heads"] == 0,
        f"{path}: heads must be divisible by kv_heads",
    )
    require(cfg["ffn_size"] >= cfg["hidden_size"], f"{path}: ffn too small")
    return cfg


def resolve_repo_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def validate_training(path: Path, repo: Path, natives: dict[Path, dict]) -> None:
    cfg = load(path)
    extra = sorted(set(cfg) - TRAIN_KEYS)
    require(not extra, f"{path}: unknown keys {extra}")
    require(cfg.get("format") == "lkjai-train-config-v1", f"{path}: bad format")
    require(cfg.get("objective") == "causal_lm_full", f"{path}: bad objective")
    if "model_kind" in cfg:
        require(cfg["model_kind"] in {"dense", "transformer", "decoder"}, f"{path}: bad kind")
    if "target_seconds" in cfg:
        require(int(cfg["target_seconds"]) > 0, f"{path}: target_seconds must be positive")
    native_path = resolve_repo_path(repo, cfg.get("native_config", ""))
    require(native_path.is_file(), f"{path}: native_config missing")
    native = natives[native_path]
    seq = int(cfg.get("sequence_len", 0))
    require(1 < seq <= native["context"], f"{path}: sequence_len exceeds context")
    for key in [
        "batch_size",
        "gradient_accumulation",
        "max_optimizer_steps",
        "save_latest_every_optimizer_steps",
    ]:
        require(int(cfg.get(key, 0)) > 0, f"{path}: {key} must be positive")
    packed = cfg.get("packed_cache_dir", "")
    if packed and not Path(packed).is_absolute():
        require((repo / packed).exists(), f"{path}: packed_cache_dir missing")


def main() -> None:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[2]
    repo = repo.resolve()
    native_paths = sorted((repo / "configs" / "native").glob("*.json"))
    training_paths = sorted((repo / "configs" / "training").glob("*.json"))
    natives = {path.resolve(): validate_native(path) for path in native_paths}
    for path in training_paths:
        validate_training(path, repo, natives)


if __name__ == "__main__":
    main()
