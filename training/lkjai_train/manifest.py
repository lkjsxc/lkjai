import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .artifacts import file_sha256, read_json


def checkpoint_manifest(paths, settings) -> Path:
    paths.ensure()
    summary = read_json(paths.training_summary) if paths.training_summary.exists() else {}
    dataset = paths.corpus if paths.corpus.exists() else paths.fixtures
    manifest = {
        "format": "lkjai-scratch-checkpoint-manifest-v1",
        "backend": summary.get("backend", "unknown"),
        "settings": asdict(settings),
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset) if dataset.exists() else "",
        "checkpoint_dir": summary.get("checkpoint_dir", ""),
        "latest_checkpoint_dir": summary.get("latest_checkpoint_dir", str(paths.checkpoint_latest) if checkpoint_exists(paths.checkpoint_latest) else ""),
        "retained_intermediate_checkpoints": [str(path) for path in retained_intermediate_checkpoints(paths)],
        "final_checkpoint_dir": summary.get("final_checkpoint_dir", ""),
        "best_checkpoint_dir": summary.get("best_checkpoint_dir", ""),
        "objective": summary.get("objective", getattr(settings, "objective", "")),
        "train_rows": summary.get("train_rows", 0),
        "metrics": summary.get("metrics", {}),
    }
    paths.checkpoint_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.checkpoint_manifest


def export_manifest(paths, settings) -> Path:
    paths.ensure()
    model_dir = paths.root.parent / "models" / settings.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(paths.tokenizer_json, model_dir / "tokenizer.json")
    checkpoint_dir = serving_checkpoint_dir(paths, settings)
    copy_if_exists(checkpoint_dir / "config.json", model_dir / "config.json")
    copy_if_exists(checkpoint_dir / "model.pt", model_dir / "model.pt")
    manifest = {
        "format": "lkjai-scratch-serving-manifest-v1",
        "model": settings.model_name,
        "tokenizer": str(model_dir / "tokenizer.json"),
        "config": str(model_dir / "config.json"),
        "checkpoint": str(model_dir / "model.pt"),
        "checkpoint_source": str(checkpoint_dir),
        "checkpoint_manifest": str(paths.checkpoint_manifest),
    }
    paths.export_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.export_manifest


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def checkpoint_exists(checkpoint_dir: Path) -> bool:
    return (
        (checkpoint_dir / "model.pt").exists()
        and (checkpoint_dir / "config.json").exists()
        and (checkpoint_dir / "training-state.pt").exists()
    )


def retained_intermediate_checkpoints(paths) -> list[Path]:
    steps = getattr(paths, "checkpoint_steps", paths.checkpoints / "steps")
    if not steps.exists():
        return []
    return sorted(path for path in steps.iterdir() if path.is_dir() and not path.name.startswith(".") and checkpoint_exists(path))


def serving_checkpoint_dir(paths, settings=None) -> Path:
    simpo_model = paths.checkpoint_simpo / "model.pt"
    dpo_model = paths.checkpoint_dpo / "model.pt"
    final_model = paths.checkpoint_final / "model.pt"
    if simpo_model.exists() and paths.simpo_summary.exists():
        summary = read_json(paths.simpo_summary)
        if summary.get("accepted") is not False:
            return paths.checkpoint_simpo
    if dpo_model.exists() and paths.dpo_summary.exists():
        summary = read_json(paths.dpo_summary)
        if summary.get("accepted") is False:
            return preferred_sft_checkpoint(paths, settings)
        if not final_model.exists() or paths.dpo_summary.stat().st_mtime >= final_model.stat().st_mtime:
            return paths.checkpoint_dpo
    return preferred_sft_checkpoint(paths, settings)


def preferred_sft_checkpoint(paths, settings=None) -> Path:
    export_checkpoint = getattr(settings, "export_checkpoint", "best")
    if export_checkpoint == "best" and (paths.checkpoint_best / "model.pt").exists():
        return paths.checkpoint_best
    return paths.checkpoint_final
