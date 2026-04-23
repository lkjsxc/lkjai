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
    copy_if_exists(paths.checkpoint_final / "config.json", model_dir / "config.json")
    copy_if_exists(paths.checkpoint_final / "model.pt", model_dir / "model.pt")
    manifest = {
        "format": "lkjai-scratch-serving-manifest-v1",
        "model": settings.model_name,
        "tokenizer": str(model_dir / "tokenizer.json"),
        "config": str(model_dir / "config.json"),
        "checkpoint": str(model_dir / "model.pt"),
        "checkpoint_manifest": str(paths.checkpoint_manifest),
    }
    paths.export_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.export_manifest


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)
