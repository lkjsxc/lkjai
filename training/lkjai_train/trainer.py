import json
from dataclasses import asdict
from pathlib import Path
import tomllib

import numpy as np
import torch

from .model import LkjModel, ModelConfig, tiny_config


def train_model(
    paths,
    tiny: bool = False,
    steps: int = 8,
    config_path: str = "",
    context: int = 0,
) -> str:
    cfg = tiny_config() if tiny else load_config(config_path)
    if context:
        cfg.context = context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.tensor(np.load(paths.tokenized / "tokens.npy"), dtype=torch.long)
    cfg.vocab_size = max(cfg.vocab_size, int(data.max().item()) + 1)
    model = LkjModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for step in range(steps):
        x, y = batch(data, cfg.context)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = paths.checkpoints / "smoke.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "losses": losses}, checkpoint)
    (paths.runs / "last-train.json").write_text(
        json.dumps({"device": device, "losses": losses}, indent=2),
        encoding="utf-8",
    )
    return str(checkpoint)


def load_config(config_path: str) -> ModelConfig:
    if not config_path:
        return ModelConfig()
    with Path(config_path).open("rb") as file:
        raw = tomllib.load(file).get("model", {})
    return ModelConfig(
        vocab_size=int(raw.get("vocab_size", 259)),
        context=int(raw.get("context", 64)),
        layers=int(raw.get("layers", 2)),
        hidden=int(raw.get("hidden", 128)),
        heads=int(raw.get("heads", 4)),
    )


def batch(data: torch.Tensor, context: int):
    if data.numel() <= context + 1:
        repeats = (context + 2) // max(1, data.numel()) + 1
        data = data.repeat(repeats)
    starts = torch.randint(0, data.numel() - context - 1, (2,))
    x = torch.stack([data[i : i + context] for i in starts])
    y = torch.stack([data[i + 1 : i + context + 1] for i in starts])
    return x, y
