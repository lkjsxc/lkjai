#!/usr/bin/env python3
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def write_cache(root: Path):
    cache = root / "datasets" / "packed" / "train-causal_lm_full-seq1024"
    cache.mkdir(parents=True)
    meta = {
        "format": "lkjai-packed-cache-v2",
        "sequence_len": 8,
        "vocab_size": 256,
        "token_dtype": "uint16",
        "row_count": 1,
        "token_count": 8,
    }
    (cache / "metadata.json").write_text(json.dumps(meta))
    (cache / "tokens.bin").write_bytes(struct.pack("<8H", *range(8)))
    (cache / "loss_mask.bin").write_bytes(bytes([1] * 8))
    (cache / "starts.bin").write_bytes(struct.pack("<Q", 0))


def main():
    train_bin, infer_bin = sys.argv[1:3]
    repo = Path(__file__).resolve().parents[2]
    root = Path("/tmp/lkjai-dense-infer")
    if root.exists():
        shutil.rmtree(root)
    write_cache(root)
    env = {**os.environ.copy(), "DATA_DIR": str(root), "MODEL_NAME": "infer-smoke"}
    subprocess.run(
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
        check=True,
        text=True,
        capture_output=True,
    )
    artifact = root / "exports" / "infer-smoke"
    result = subprocess.run(
        [infer_bin, "--model-dir", str(artifact), "--tokens", "1,2,3"],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["model_kind"] == "dense"
    assert payload["shape"] == [1, 256]
    assert payload["finite"] is True
    assert payload["checksum"]
    assert isinstance(payload["top_token"], int)
    bad = subprocess.run(
        [infer_bin, "--model-dir", str(artifact), "--tokens", "1,999"],
        text=True,
        capture_output=True,
    )
    assert bad.returncode != 0
    assert "outside dense model vocab" in bad.stderr
    missing = subprocess.run(
        [infer_bin, "--model-dir", str(root / "missing"), "--tokens", "1"],
        text=True,
        capture_output=True,
    )
    assert missing.returncode != 0


if __name__ == "__main__":
    main()
