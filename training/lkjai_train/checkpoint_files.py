import json
import os
import shutil
import time
from pathlib import Path


STATE_FILE = "training-state.pt"
METADATA_FILE = "metadata.json"
SNAPSHOT_ROOT = ".snapshots"


def checkpoint_exists(checkpoint_dir: Path) -> bool:
    return (
        (checkpoint_dir / "model.pt").exists()
        and (checkpoint_dir / "config.json").exists()
        and (checkpoint_dir / STATE_FILE).exists()
    )


def latest_complete_checkpoint(paths, source: str = "latest") -> Path | None:
    for candidate in checkpoint_candidates(paths, source):
        if checkpoint_exists(candidate):
            return candidate
    return None


def write_checkpoint_manifest(paths, settings) -> Path:
    paths.ensure()
    manifest = {
        "format": "lkjai-checkpoint-manifest-v2",
        "latest_checkpoint_dir": str(paths.checkpoint_latest) if checkpoint_exists(paths.checkpoint_latest) else "",
        "retained_intermediate_checkpoints": [str(path) for path in retained_intermediate_checkpoints(paths)],
        "best_checkpoint_dir": str(paths.checkpoint_best) if checkpoint_exists(paths.checkpoint_best) else "",
        "final_checkpoint_dir": str(paths.checkpoint_final) if checkpoint_exists(paths.checkpoint_final) else "",
        "settings": getattr(settings, "__dict__", {}),
    }
    paths.checkpoint_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.checkpoint_manifest


def prune_old_checkpoints(paths, keep_last: int) -> list[Path]:
    if keep_last < 0:
        raise ValueError("TRAIN_KEEP_LAST_CHECKPOINTS must be >= 0")
    checkpoints = retained_intermediate_checkpoints(paths)
    remove = checkpoints[: max(0, len(checkpoints) - keep_last)]
    for path in remove:
        shutil.rmtree(path, ignore_errors=True)
    return retained_intermediate_checkpoints(paths)


def checkpoint_candidates(paths, source: str) -> list[Path]:
    if source not in {"latest", "final", "best"}:
        raise ValueError("TRAIN_CHECKPOINT_RESUME_SOURCE must be latest, final, or best")
    latest = getattr(paths, "checkpoint_latest", paths.checkpoints / "latest")
    return {
        "latest": [latest, paths.checkpoint_final],
        "final": [paths.checkpoint_final],
        "best": [paths.checkpoint_best],
    }[source]


def retained_intermediate_checkpoints(paths) -> list[Path]:
    steps = getattr(paths, "checkpoint_steps", paths.checkpoints / "steps")
    if not steps.exists():
        return []
    checkpoints = [path for path in steps.iterdir() if path.is_dir() and not path.name.startswith(".") and checkpoint_exists(path)]
    return sorted(checkpoints, key=checkpoint_sort_key)


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    metadata_path = path / METADATA_FILE
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return int(metadata.get("optimizer_steps", 0)), path.name
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0, path.name


def snapshot_target(checkpoint_dir: Path, source_type: str, counters: dict) -> Path:
    if is_alias_checkpoint(checkpoint_dir):
        root = checkpoint_dir.parent / SNAPSHOT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        step = int(counters.get("optimizer_steps", 0))
        stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        return root / f"{source_type}-step-{step:06d}-{stamp}"
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def is_alias_checkpoint(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.name in {"best", "final", "latest"}


def promote_directory(temp_dir: Path, target: Path) -> None:
    if os.path.lexists(target):
        backup = target.parent / f".{target.name}.replaced-{int(time.time() * 1000)}"
        os.replace(target, backup)
        os.replace(temp_dir, target)
        remove_replaced(backup)
        return
    os.replace(temp_dir, target)


def remove_replaced(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def replace_alias(alias: Path, target: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    old_target = alias.resolve() if alias.is_symlink() else None
    temp_link = alias.parent / f".{alias.name}.link.tmp-{os.getpid()}"
    if temp_link.exists() or temp_link.is_symlink():
        temp_link.unlink()
    os.symlink(os.path.relpath(target, alias.parent), temp_link, target_is_directory=True)
    if os.path.lexists(alias):
        replace_existing_alias(alias, temp_link)
    else:
        os.replace(temp_link, alias)
    if old_target and old_target.parent.name == SNAPSHOT_ROOT and old_target != target:
        shutil.rmtree(old_target, ignore_errors=True)


def replace_existing_alias(alias: Path, temp_link: Path) -> None:
    if alias.is_symlink() or alias.is_file():
        os.replace(temp_link, alias)
        return
    backup = alias.parent / f".{alias.name}.old-{int(time.time() * 1000)}"
    os.replace(alias, backup)
    os.replace(temp_link, alias)
    shutil.rmtree(backup, ignore_errors=True)
