import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import file_sha256


def train_adapter_manifest(paths, settings) -> Path:
    paths.ensure()
    manifest = {
        "format": "lkjai-adapter-manifest-v1",
        "backend": "qlora",
        "settings": asdict(settings),
        "dataset": str(paths.fixtures),
        "dataset_sha256": file_sha256(paths.fixtures) if paths.fixtures.exists() else "",
        "note": "Run Axolotl or an equivalent QLoRA backend with these settings for full tuning.",
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
    }
    paths.export_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.export_manifest
