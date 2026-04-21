import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from .model import LkjModel, ModelConfig


def export_model(paths, max_mib: int = 512) -> Path:
    checkpoint = torch.load(paths.checkpoints / "smoke.pt", map_location="cpu")
    cfg = ModelConfig(**checkpoint["config"])
    model = LkjModel(cfg)
    model.load_state_dict(checkpoint["model"])
    out = paths.models / "lkj-150m"
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "model.safetensors"
    state = {key: value.half() for key, value in model.state_dict().items()}
    save_file(state, weights)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    size = sum(path.stat().st_size for path in out.glob("*") if path.is_file())
    size_mib = size / 1024 / 1024
    if size_mib > max_mib:
        raise RuntimeError(f"export is {size_mib:.2f} MiB > {max_mib} MiB")
    (out / "size.json").write_text(json.dumps({"size_mib": size_mib}, indent=2), encoding="utf-8")
    return out
