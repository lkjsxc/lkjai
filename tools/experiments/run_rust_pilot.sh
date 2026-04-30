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

docker build -f "$ROOT/ops/docker/Dockerfile.native" -t lkjai-native-pilot "$ROOT" >/dev/null
docker run --rm --entrypoint lkjai-native-train lkjai-native-pilot --smoke --steps 2 \
  | tee "$OUT_DIR/native-train-smoke.json"

echo "{\"run_id\":\"$RUN_ID\",\"out_dir\":\"$OUT_DIR\",\"status\":\"pass\"}"
