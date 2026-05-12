#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-decoder-2h-$(date +%Y%m%d-%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-decoder-2h-40m-3070}"
TARGET_SECONDS="${TARGET_SECONDS:-7200}"
SEQ_LEN="${SEQ_LEN:-1024}"

echo "[1/5] Build native and runtime images"
docker compose --profile inference --profile web --profile train build

echo "[2/5] Run decoder benchmark"
PHASE="smoke"
TRAIN_ARGS=(--train --mode decoder --max-steps "${SMOKE_STEPS:-2}")
if [[ "${REQUIRE_ACCEPTED_CUDA:-0}" == "1" ]]; then
  PHASE="full"
  TRAIN_ARGS=(--train --mode decoder --target-seconds "$TARGET_SECONDS")
fi
TRAIN_DATA_DIR="/app/data/perf-runs/$RUN_ID/decoder_2h_bf16_cuda/$PHASE" \
MODEL_NAME="$MODEL_NAME" \
docker compose --profile train run --rm train \
  "${TRAIN_ARGS[@]}" \
  --config /workspace/configs/native/decoder_40m_bf16_3070.json \
  --tokenizer /app/data/train/tokenizer/tokenizer.json \
  --packed-cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len "$SEQ_LEN" \
  --run-purpose decoder_2h_demo

REPORT="data/perf-runs/$RUN_ID/decoder_2h_bf16_cuda/$PHASE/runs/train-report.json"
test -f "$REPORT"
if [[ "${REQUIRE_ACCEPTED_CUDA:-0}" == "1" ]]; then
  grep -q '"accepted_cuda_training":true' "$REPORT"
  grep -q '"implementation_status":"accepted"' "$REPORT"
  grep -q '"decoder_cuda_slice":"full_decoder"' "$REPORT"
  grep -q '"decoder_backward_backend":"cuda_full_decoder"' "$REPORT"
  grep -q '"kv_cache_backend":"cuda_contiguous_bf16"' "$REPORT"
  grep -q '"decode_backend":"cuda_kv_cache"' "$REPORT"
fi

echo "[3/5] Publish model artifact"
SRC="data/perf-runs/$RUN_ID/decoder_2h_bf16_cuda/$PHASE/exports/$MODEL_NAME"
test -d "$SRC"
rm -rf "data/models/$MODEL_NAME"
mkdir -p data/models
cp -R "$SRC" "data/models/$MODEL_NAME"

echo "[4/5] Start services"
MODEL_NAME="$MODEL_NAME" docker compose --profile inference up -d inference
MODEL_NAME="$MODEL_NAME" docker compose --profile web up -d web

echo "[5/5] Health checks"
curl --fail http://127.0.0.1:${MODEL_PORT:-8081}/healthz
curl --fail http://127.0.0.1:${MODEL_PORT:-8081}/v1/models
curl --fail http://127.0.0.1:${APP_PORT:-8080}/healthz
printf 'Run ID: %s\nModel: %s\nWeb: http://127.0.0.1:%s\n' \
  "$RUN_ID" "$MODEL_NAME" "${APP_PORT:-8080}"
