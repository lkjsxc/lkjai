import json
import time
from dataclasses import asdict
from pathlib import Path
import tomllib

import numpy as np
import torch

from .artifacts import file_sha256
from .model import LkjModel, ModelConfig, tiny_config


def train_model(
    paths,
    tiny: bool = False,
    steps: int = 8,
    config_path: str = "",
    context: int = 0,
    max_duration_secs: int = 0,
    checkpoint_name: str = "latest.pt",
) -> str:
    cfg = tiny_config() if tiny else load_config(config_path)
    if context:
        cfg.context = context
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, metadata = load_tokens(paths)
    verify_tokenizer_hash(paths, metadata)
    if data.size == 0:
        raise ValueError("tokenized corpus is empty")
    max_token_id = int(metadata.get("max_token_id", int(data.max())))
    cfg.vocab_size = max(cfg.vocab_size, max_token_id + 1)
    model = LkjModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    min_steps = max(1, int(steps))
    duration_target = max(0, int(max_duration_secs))
    start = time.monotonic()
    deadline = start + duration_target if duration_target else None
    step = 0
    while step < min_steps or (deadline is not None and time.monotonic() < deadline):
        x, y = batch(data, cfg.context)
        x = torch.from_numpy(x).to(device=device, dtype=torch.long)
        y = torch.from_numpy(y).to(device=device, dtype=torch.long)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        step += 1
    duration_secs = time.monotonic() - start
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = paths.checkpoints / checkpoint_name
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "losses": losses}, checkpoint)
    stop_reason = "steps-or-duration" if duration_target else "steps"
    (paths.runs / "last-train.json").write_text(
        json.dumps(
            {
                "device": device,
                "losses": losses,
                "steps_completed": step,
                "duration_secs": duration_secs,
                "min_steps": min_steps,
                "duration_target_secs": duration_target,
                "stop_reason": stop_reason,
            },
            indent=2,
        ),
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


def load_tokens(paths):
    metadata_path = paths.tokenized / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        token_file = metadata.get("token_file")
        if token_file:
            count = int(metadata.get("tokens", 0))
            token_path = paths.tokenized / token_file
            data = np.memmap(token_path, dtype=np.uint16, mode="r", shape=(count,))
            return data, metadata
    legacy = np.load(paths.tokenized / "tokens.npy", mmap_mode="r")
    metadata = {"tokens": int(legacy.size)}
    if legacy.size:
        metadata["max_token_id"] = int(legacy.max())
    return legacy, metadata


def verify_tokenizer_hash(paths, metadata) -> None:
    expected = metadata.get("tokenizer_sha256")
    if not expected:
        raise ValueError("tokenized corpus metadata is missing tokenizer_sha256; rerun pack-tokens")
    tokenizer = paths.tokenizers / "tokenizer.json"
    if not tokenizer.exists():
        raise ValueError(f"tokenizer missing: {tokenizer}")
    actual = file_sha256(tokenizer)
    if actual != expected:
        raise ValueError("tokenizer does not match packed tokens; rerun pack-tokens")


def batch(data, context: int, batch_size: int = 2):
    context = max(1, int(context))
    length = int(data.shape[0])
    if length <= context + 1:
        if length == 0:
            raise ValueError("tokenized corpus is empty")
        repeats = (context + 2) // length + 1
        data = np.tile(np.asarray(data, dtype=np.uint16), repeats)
        length = int(data.shape[0])
    starts = np.random.randint(0, length - context - 1, size=batch_size)
    offsets = starts[:, None] + np.arange(context, dtype=np.int64)[None, :]
    x = np.asarray(data[offsets], dtype=np.int64)
    y = np.asarray(data[offsets + 1], dtype=np.int64)
    return x, y
