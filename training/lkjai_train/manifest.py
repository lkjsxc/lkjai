import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import file_sha256, read_json


def train_adapter_manifest(paths, settings) -> Path:
    paths.ensure()
    summary = read_json(paths.training_summary) if paths.training_summary.exists() else {}
    manifest = {
        "format": "lkjai-adapter-manifest-v1",
        "backend": summary.get("backend", "unknown"),
        "settings": asdict(settings),
        "dataset": str(paths.fixtures),
        "dataset_sha256": file_sha256(paths.fixtures) if paths.fixtures.exists() else "",
        "checkpoint_dir": summary.get("checkpoint_dir", ""),
        "train_rows": summary.get("train_rows", 0),
        "eval_rows": summary.get("eval_rows", 0),
        "metrics": summary.get("metrics", {}),
    }
    paths.adapter_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.adapter_manifest


def export_manifest(paths, settings) -> Path:
    paths.ensure()
    manifest = {
        "format": "lkjai-gguf-export-manifest-v1",
        "serving_model": "qwen3-1.7b-q4",
        "tuning_base_model": settings.base_model,
        "quantization": "Q4_K_M",
        "adapter_manifest": str(paths.adapter_manifest),
        "adapter_dir": str(paths.adapter_final),
    }
    paths.export_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.export_manifest
