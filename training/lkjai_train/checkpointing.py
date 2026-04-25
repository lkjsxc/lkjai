import random
import shutil
from pathlib import Path

import torch

from .scratch_model import save_config


STATE_FILE = "training-state.pt"


def checkpoint_exists(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "model.pt").exists() and (checkpoint_dir / STATE_FILE).exists()


def save_checkpoint(
    checkpoint_dir: Path,
    config,
    model,
    optimizer,
    scheduler,
    scaler,
    counters: dict,
    settings,
    best_metric: float,
    validation_history: list[dict],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), checkpoint_dir / "model.pt")
    save_config(config, checkpoint_dir / "config.json")
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "counters": counters,
        "best_metric": best_metric,
        "validation_history": validation_history,
        "settings": settings.__dict__,
    }
    torch.save(state, checkpoint_dir / STATE_FILE)


def load_checkpoint(checkpoint_dir: Path, model, optimizer, scheduler, scaler, device) -> dict:
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True))
    state = torch.load(checkpoint_dir / STATE_FILE, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng_state(state.get("rng", {}))
    return state


def copy_serving_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["model.pt", "config.json", STATE_FILE]:
        path = src / name
        if path.exists():
            shutil.copyfile(path, dst / name)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)
