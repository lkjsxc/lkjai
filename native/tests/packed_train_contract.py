#!/usr/bin/env python3
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


def write_cache(root: Path):
    cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "format": "lkjai-packed-cache-v2",
                "split": "train",
                "objective": "causal_lm_full",
                "sequence_len": 1024,
                "vocab_size": 8192,
                "token_dtype": "uint16",
                "row_count": 1,
                "token_count": 8,
            }
        )
    )
    (cache / "tokens.bin").write_bytes(struct.pack("<8H", *range(8)))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * 8))
    (cache / "starts.bin").write_bytes(struct.pack("<Q", 0))


def main():
    train_bin = sys.argv[1]
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
        [train_bin, "--train"], env=env, text=True, capture_output=True, check=True
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass", result.stdout
    manifest = root / "exports" / "packed-smoke" / "manifest.json"
    assert "lkjai-native-artifact-v2" in manifest.read_text()


if __name__ == "__main__":
    main()
