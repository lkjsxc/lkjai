#!/usr/bin/env python3
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def write_cache(root: Path, version: str = "lkjai-packed-cache-v2"):
    cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "format": version,
                "split": "train",
                "objective": "causal_lm_full",
                "sequence_len": 8,
                "vocab_size": 256,
                "token_dtype": "uint16",
                "row_count": 1,
                "token_count": 8,
            }
        )
    )
    (cache / "tokens.bin").write_bytes(struct.pack("<8H", *range(8)))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * 8))
    (cache / "starts.bin").write_bytes(struct.pack("<Q", 0))


def check_schema(artifact: Path, inspect_bin: str):
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["format"] == "lkjai-native-artifact-v2"
    assert manifest["kind"] == "dense"
    assert manifest["artifact_kind"] == "export"
    assert manifest["weights_checksum"]
    assert manifest["config_checksum"]
    assert manifest["tokenizer_checksum"]
    index = json.loads((artifact / "weights.index.json").read_text())
    names = {tensor["name"] for tensor in index["tensors"]}
    assert {"tok_embeddings", "lm_head"} <= names
    first = index["tensors"][0]
    assert {"name", "dtype", "shape", "byte_offset", "byte_length"} <= set(first)
    checkpoint = artifact.parents[1] / "checkpoints" / "latest"
    ckpt_manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert ckpt_manifest["kind"] == "dense"
    assert ckpt_manifest["artifact_kind"] == "checkpoint"
    assert (checkpoint / "optimizer.index.json").is_file()
    opt_names = {
        tensor["name"]
        for tensor in json.loads((checkpoint / "optimizer.index.json").read_text())[
            "tensors"
        ]
    }
    assert {
        "master.tok_embeddings",
        "adam_m.tok_embeddings",
        "adam_v.tok_embeddings",
        "master.lm_head",
        "adam_m.lm_head",
        "adam_v.lm_head",
    } <= opt_names
    broken = artifact.parent / "broken-schema"
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(artifact, broken)
    bad = dict(manifest)
    bad["tokenizer_checksum"] = "bad"
    (broken / "manifest.json").write_text(json.dumps(bad))
    result = subprocess.run(
        [inspect_bin, "--model-dir", str(broken)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "checksum" in result.stderr


def main():
    train_bin, logits_bin, migrate_bin, inspect_bin = sys.argv[1:5]
    repo = Path(__file__).resolve().parents[2]
    root = Path("/tmp/lkjai-packed-train")
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    write_cache(root)
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(root),
            "MODEL_NAME": "packed-smoke",
            "TRAIN_MAX_OPTIMIZER_STEPS": "2",
            "TRAIN_LOG_EVERY_OPTIMIZER_STEPS": "1",
            "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "1",
        }
    )
    result = subprocess.run(
        [
            train_bin,
            "--train",
            "--config",
            str(repo / "configs" / "native" / "native_debug_bf16.json"),
            "--seq-len",
            "8",
            "--max-steps",
            "2",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass" and payload["dense_cuda_path"] is True
    assert "transformer_path" not in payload, result.stdout
    assert payload["initial_loss"] > payload["loss"], result.stdout
    assert payload["loss_finite"] is True, result.stdout
    assert payload["weight_changed"] is True, result.stdout
    assert payload["logits_checksum"], result.stdout
    for key in ["batch_load", "forward", "backward", "optimizer", "checkpoint", "export"]:
        assert key in payload["timings"], result.stdout
    manifest = root / "exports" / "packed-smoke" / "manifest.json"
    assert "lkjai-native-artifact-v2" in manifest.read_text()
    artifact = root / "exports" / "packed-smoke"
    subprocess.run([inspect_bin, "--model-dir", str(artifact)], check=True)
    check_schema(artifact, inspect_bin)
    result = subprocess.run(
        [logits_bin, "--model-dir", str(artifact), "--tokens", "1,2,3"],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["finite"] is True, result.stdout
    assert payload["shape"] == [1, 256], result.stdout
    result = subprocess.run(
        [
            train_bin,
            "--train",
            "--config",
            str(repo / "configs" / "native" / "native_debug_bf16.json"),
            "--seq-len",
            "8",
            "--max-steps",
            "1",
            "--resume",
            str(root / "checkpoints" / "latest"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["start_step"] == 2, result.stdout
    assert payload["steps"] == 3, result.stdout
    v1 = root / "v1"
    write_cache(v1, "lkjai-packed-cache-v1")
    v2 = root / "v2"
    subprocess.run(
        [
            migrate_bin,
            "--migrate-v1-to-v2",
            "--in",
            str(v1 / "datasets" / "packed" / "train-causal_lm_full-seq1024"),
            "--out",
            str(v2),
            "--config",
            str(repo / "configs" / "native" / "native_debug_bf16.json"),
            "--link-mode",
            "hardlink",
        ],
        check=True,
    )
    assert "lkjai-packed-cache-v2" in (v2 / "metadata.json").read_text()
    result = subprocess.run(
        [
            train_bin,
            "--train",
            "--packed-cache",
            str(v2),
            "--config",
            str(repo / "configs" / "native" / "native_debug_bf16.json"),
            "--seq-len",
            "8",
            "--max-steps",
            "1",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dense_cuda_path"] is True and "transformer_path" not in payload


if __name__ == "__main__":
    main()
