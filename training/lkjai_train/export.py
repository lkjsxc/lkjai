import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from .artifacts import file_sha256, read_json
from .model import LkjModel, ModelConfig


def export_model(paths, max_mib: int = 512) -> Path:
    checkpoint = torch.load(paths.checkpoints / "latest.pt", map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["config"])
    model = LkjModel(cfg)
    model.load_state_dict(checkpoint["model"])
    out = paths.models / "lkj-150m"
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "model.safetensors"
    state = {key: value.half() for key, value in model.state_dict().items()}
    save_file(state, weights)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    tokenizer = paths.tokenizers / "tokenizer.json"
    if not tokenizer.exists():
        raise ValueError(f"tokenizer missing: {tokenizer}")
    verify_tokenizer_matches_tokens(paths, tokenizer)
    shutil.copy2(tokenizer, out / "tokenizer.json")
    write_manifest(out, cfg, tokenizer, weights)
    size = sum(path.stat().st_size for path in out.glob("*") if path.is_file())
    size_mib = size / 1024 / 1024
    if size_mib > max_mib:
        raise RuntimeError(f"export is {size_mib:.2f} MiB > {max_mib} MiB")
    (out / "size.json").write_text(json.dumps({"size_mib": size_mib}, indent=2), encoding="utf-8")
    return out


def verify_tokenizer_matches_tokens(paths, tokenizer: Path) -> None:
    metadata = paths.tokenized / "metadata.json"
    if not metadata.exists():
        raise ValueError("tokenized corpus metadata missing; rerun pack-tokens")
    expected = read_json(metadata).get("tokenizer_sha256")
    if not expected:
        raise ValueError("tokenized corpus metadata is missing tokenizer_sha256; rerun pack-tokens")
    actual = file_sha256(tokenizer)
    if actual != expected:
        raise ValueError("tokenizer does not match packed tokens; rerun pack-tokens")


def write_manifest(out: Path, cfg: ModelConfig, tokenizer: Path, weights: Path) -> None:
    manifest = {
        "format": "lkjai-export-v1",
        "config": "config.json",
        "weights": "model.safetensors",
        "tokenizer": "tokenizer.json",
        "vocab_size": cfg.vocab_size,
        "context": cfg.context,
        "tokenizer_sha256": file_sha256(tokenizer),
        "weights_sha256": file_sha256(weights),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
