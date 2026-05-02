#!/usr/bin/env python3
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def write_cache(root: Path) -> Path:
    cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    rows = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]
    flat = [token for row in rows for token in row]
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "format": "lkjai-packed-cache-v2",
                "split": "train",
                "objective": "causal_lm_full",
                "sequence_len": 8,
                "vocab_size": 256,
                "token_dtype": "uint16",
                "row_count": len(rows),
                "token_count": len(flat),
            }
        )
    )
    (cache / "tokens.bin").write_bytes(struct.pack("<" + "H" * len(flat), *flat))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * len(flat)))
    (cache / "starts.bin").write_bytes(struct.pack("<2Q", 0, 8))
    return cache


def run_train(train_bin: str, root: Path, repo: Path, steps: int, extra=None):
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(root),
            "MODEL_NAME": "transformer-smoke",
            "TRAIN_MAX_OPTIMIZER_STEPS": str(steps),
            "TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS": "1",
        }
    )
    cmd = [
        train_bin,
        "--train",
        "--mode",
        "transformer",
        "--config",
        str(repo / "configs" / "native" / "native_transformer_debug_bf16.json"),
        "--seq-len",
        "8",
        "--max-steps",
        str(steps),
        "--lr",
        "0.01",
    ]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    persisted = json.loads((root / "runs" / "train-report.json").read_text())
    assert persisted["schema_version"] == payload["schema_version"] == 3
    assert persisted["model_kind"] == payload["model_kind"] == "transformer"
    assert persisted["accepted_cuda_training"] is False
    assert persisted["implementation_status"] == "experimental"
    assert persisted["transformer_status"] == "experimental"
    assert persisted["transformer_cuda_path"] is False
    assert persisted["transformer_cuda_probe"] is True
    assert persisted["forward_backend"] == "host_reference"
    assert persisted["backward_backend"] == "host_surrogate"
    assert persisted["optimizer_backend"] == "host_adamw_fp32"
    assert persisted["logits_check"]["status"] == "pass"
    return payload


def optimizer_names(checkpoint: Path) -> set[str]:
    return {
        tensor["name"]
        for tensor in json.loads((checkpoint / "optimizer.index.json").read_text())[
            "tensors"
        ]
    }


def main():
    train_bin, logits_bin, inspect_bin = sys.argv[1:4]
    repo = Path(__file__).resolve().parents[2]
    root = Path("/tmp/lkjai-transformer-train")
    if root.exists():
        shutil.rmtree(root)
    write_cache(root)
    payload = run_train(train_bin, root, repo, 3)
    assert payload["status"] == "success", payload
    assert payload["initial_loss"] > payload["loss"], payload
    assert payload["weight_changed"] is True, payload
    artifact = root / "exports" / "transformer-smoke"
    subprocess.run([inspect_bin, "--model-dir", str(artifact)], check=True)
    logits = subprocess.run(
        [logits_bin, "--model-dir", str(artifact), "--tokens", "1,2,3"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(logits.stdout)["shape"] == [1, 256]
    names = optimizer_names(root / "checkpoints" / "latest")
    for base in ["tok_embeddings", "pos_embeddings", "lm_head"]:
        assert {f"master.{base}", f"adam_m.{base}", f"adam_v.{base}"} <= names
    resumed = run_train(
        train_bin,
        root,
        repo,
        1,
        ["--resume", str(root / "checkpoints" / "latest")],
    )
    assert resumed["start_step"] == 3
    assert resumed["steps"] == 4
    bad = subprocess.run(
        [
            train_bin,
            "--train",
            "--mode",
            "transformer",
            "--config",
            str(repo / "configs" / "native" / "native_debug_bf16.json"),
            "--seq-len",
            "8",
            "--max-steps",
            "1",
        ],
        env={**os.environ.copy(), "DATA_DIR": str(root)},
        text=True,
        capture_output=True,
    )
    assert bad.returncode != 0
    assert "tie_embeddings=false" in bad.stderr


if __name__ == "__main__":
    main()
