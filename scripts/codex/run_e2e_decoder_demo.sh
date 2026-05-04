#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-decoder-2h-$(date +%Y%m%d-%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-decoder-2h-18m-3070}"
TARGET_SECONDS="${TARGET_SECONDS:-7200}"
SEQ_LEN="${SEQ_LEN:-1024}"

echo "[1/5] Build native and runtime images"
docker compose build inference web train

echo "[2/5] Run decoder benchmark"
RUNNER_MODE=(--smoke-steps "${SMOKE_STEPS:-2}")
if [[ "${REQUIRE_ACCEPTED_CUDA:-0}" == "1" ]]; then
  RUNNER_MODE=(--full --require-accepted-cuda)
fi
python3 tools/benchmarks/run_decoder_2h.py \
  --run-id "$RUN_ID" \
  --native-config configs/native/decoder_18m_bf16_3070.json \
  --source data/train/datasets/train.jsonl \
  --tokenizer data/train/tokenizer/tokenizer.json \
  --cache data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len "$SEQ_LEN" \
  --target-seconds "$TARGET_SECONDS" \
  --model-name "$MODEL_NAME" \
  "${RUNNER_MODE[@]}"

echo "[3/5] Publish model artifact"
PHASE="smoke"
if [[ "${REQUIRE_ACCEPTED_CUDA:-0}" == "1" ]]; then
  PHASE="full"
fi
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
