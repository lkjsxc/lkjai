#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="$ROOT/artifacts/experiments/rust-pilot-$RUN_ID"
mkdir -p "$OUT_DIR"

docker run --rm -i \
  -v "$ROOT:/workspace" \
  -w /workspace/tools/packed-reader \
  debian:bookworm-slim \
  bash -lc 'apt-get update >/dev/null && apt-get install -y --no-install-recommends cargo ca-certificates pkg-config >/dev/null && cargo run --release -- --cache-dir /workspace/data/train/datasets/packed/train-causal_lm_full-seq1024 --sequence-len 1024 --windows 20000' \
  | tee "$OUT_DIR/rust-packed-reader.json"

docker run --rm -i \
  -v "$ROOT:/workspace" \
  -w /workspace \
  --entrypoint python \
  lkjai-train-perf:manual \
  - <<'PY' | tee "$OUT_DIR/python-packed-reader.json"
import json
import time
from pathlib import Path

from lkjai_train.packed_data import PackedDataset

cache = Path("data/train/datasets/packed/train-causal_lm_full-seq1024")
dataset = PackedDataset(cache, 1024, 0)
windows = min(20000, len(dataset))
started = time.perf_counter()
checksum = 0
for index in range(windows):
    input_ids, labels = dataset[index]
    checksum = (checksum + int(input_ids.sum()) + int(labels[labels != -100].sum())) & ((1 << 64) - 1)
elapsed = max(1e-9, time.perf_counter() - started)
print(json.dumps({
    "cache_dir": str(cache),
    "sequence_len": 1024,
    "windows_read": windows,
    "elapsed_seconds": elapsed,
    "windows_per_second": windows / elapsed,
    "tokens_per_second": windows * 1024 / elapsed,
    "checksum": checksum,
}, indent=2))
PY

echo "{\"run_id\":\"$RUN_ID\",\"out_dir\":\"$OUT_DIR\",\"status\":\"pass\"}"
