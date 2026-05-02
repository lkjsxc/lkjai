#!/usr/bin/env python3
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


def write_cache(root: Path, version: str) -> Path:
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
    return cache


def main() -> None:
    train_bin, migrate_bin = sys.argv[1:3]
    repo = Path(__file__).resolve().parents[2]
    root = Path("/tmp/lkjai-packed-migration")
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    source = write_cache(root / "v1", "lkjai-packed-cache-v1")
    migrated = root / "v2"
    config = repo / "configs" / "native" / "native_debug_bf16.json"
    subprocess.run(
        [
            migrate_bin,
            "--migrate-v1-to-v2",
            "--in",
            str(source),
            "--out",
            str(migrated),
            "--config",
            str(config),
            "--link-mode",
            "hardlink",
        ],
        check=True,
    )
    assert "lkjai-packed-cache-v2" in (migrated / "metadata.json").read_text()
    env = {**os.environ.copy(), "DATA_DIR": str(root), "MODEL_NAME": "migrated"}
    result = subprocess.run(
        [
            train_bin,
            "--train",
            "--packed-cache",
            str(migrated),
            "--config",
            str(config),
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
