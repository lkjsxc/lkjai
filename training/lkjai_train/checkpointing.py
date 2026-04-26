import json
import shutil
import tempfile
import time
from pathlib import Path

import torch

from .checkpoint_files import (
    METADATA_FILE,
    STATE_FILE,
    checkpoint_exists,
    is_alias_checkpoint,
    latest_complete_checkpoint,
    promote_directory,
    prune_old_checkpoints,
    replace_alias,
    retained_intermediate_checkpoints,
    snapshot_target,
    write_checkpoint_manifest,
)
from .checkpoint_state import restore_rng_state, rng_state, unwrap_model
from .scratch_model import save_config


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
    source_type = checkpoint_dir.name if checkpoint_dir.name in {"best", "final", "latest"} else "intermediate"
    save_checkpoint_atomic(
        checkpoint_dir,
        config,
        model,
        optimizer,
        scheduler,
        scaler,
        counters,
        settings,
        best_metric,
        validation_history,
        source_type=source_type,
    )


def save_checkpoint_atomic(
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
    source_type: str = "latest",
    validation_loss: float | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    target = snapshot_target(checkpoint_dir, source_type, counters)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        write_checkpoint_contents(
            temp_dir,
            config,
            model,
            optimizer,
            scheduler,
            scaler,
            counters,
            settings,
            best_metric,
            validation_history,
            source_type,
            validation_loss,
        )
        promote_directory(temp_dir, target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    if is_alias_checkpoint(checkpoint_dir):
        replace_alias(checkpoint_dir, target)
    return checkpoint_dir if is_alias_checkpoint(checkpoint_dir) else target


def load_checkpoint(checkpoint_dir: Path, model, optimizer, scheduler, scaler, device) -> dict:
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True))
    state = torch.load(checkpoint_dir / STATE_FILE, map_location=device, weights_only=False)
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except (KeyError, TypeError, ValueError):
            state["scheduler_load_warning"] = "checkpoint scheduler state was incompatible"
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    try:
        restore_rng_state(state.get("rng", {}))
    except (TypeError, ValueError, RuntimeError) as error:
        state["rng_load_warning"] = f"checkpoint RNG state was incompatible: {error}"
    return state


def copy_serving_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["model.pt", "config.json", STATE_FILE]:
        path = src / name
        if path.exists():
            shutil.copyfile(path, dst / name)


def write_checkpoint_contents(
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
    source_type: str,
    validation_loss: float | None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), checkpoint_dir / "model.pt")
    save_config(config, checkpoint_dir / "config.json")
    torch.save(training_state(optimizer, scheduler, scaler, counters, settings, best_metric, validation_history), checkpoint_dir / STATE_FILE)
    metadata = checkpoint_metadata(counters, best_metric, validation_history, source_type, validation_loss)
    (checkpoint_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def training_state(optimizer, scheduler, scaler, counters, settings, best_metric, validation_history) -> dict:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "counters": dict(counters),
        "best_metric": best_metric,
        "validation_history": list(validation_history),
        "settings": getattr(settings, "__dict__", {}),
    }


def checkpoint_metadata(counters: dict, best_metric: float, history: list[dict], source_type: str, validation_loss: float | None) -> dict:
    if validation_loss is None and history:
        validation_loss = history[-1].get("loss")
    return {
        "optimizer_steps": int(counters.get("optimizer_steps", 0)),
        "microsteps": int(counters.get("microsteps", 0)),
        "validation_loss": validation_loss,
        "best_metric": best_metric,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_type": source_type,
    }
