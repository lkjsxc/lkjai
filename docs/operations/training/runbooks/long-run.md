# Scratch Training Run Contract

Owner: `docs/operations/training/runbooks/long-run.md`.
State: canonical operator run contract.

## Goal

Run one measurable long training job for the 3070-first decoder 40M model.

## Default Behavior

- `docker compose --profile train up --build train` runs the committed decoder
  40M profile with a two-hour wall-clock target.
- `TRAIN_CONFIG=/workspace/configs/training/decoder_2h_40m_3070.json` is
  the default training-run config.
- `TRAIN_NATIVE_CONFIG=/workspace/configs/native/decoder_40m_bf16_3070.json`
  is the default model-shape config for the train service.
- `TRAIN_MODEL_NAME=decoder-40m-3070` is the default training export name
  and is independent from serving `MODEL_NAME`.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.
- Training defaults to `TRAIN_OBJECTIVE=causal_lm_full`.
- The current native trainer consumes an existing packed cache; tokenizer and
  cache construction are separate operations.
- Validate or rebuild
  `data/train/datasets/packed/train-causal_lm_full-seq1024` before any long
  decoder run; this path previously held a stale seq16/vocab256 smoke cache.
- Export writes the final decoder artifact for the run.
- Resume is explicit through `--resume DIR` and restores the dense FP32 master
  weights plus Adam moments from the checkpoint optimizer artifact.
- The current native binary supports two modes:
  `lkjai-native-train --smoke --steps N` and `lkjai-native-train --train`.
  Compose uses `--train --mode decoder` as the service default. Smoke checks must
  override the command explicitly.

## Supported Native Knobs

- `TRAIN_CONFIG`: training-run JSON config.
- `TRAIN_NATIVE_CONFIG`: native model-shape JSON config; `--config` overrides it.
- `TRAIN_MODEL_NAME`: Compose-only training artifact name; maps to container
  `MODEL_NAME` and defaults to `decoder-40m-3070`.
- `TRAIN_SEQUENCE_LEN`: sequence length; `--seq-len` overrides it.
- `TRAIN_BATCH_SIZE`: microbatch size; `--batch-size` overrides it.
- `TRAIN_GRADIENT_ACCUMULATION`: microsteps per AdamW step; `--grad-accum`
  overrides it.
- `TRAIN_MAX_STEPS` or `TRAIN_MAX_OPTIMIZER_STEPS`: optimizer-step cap;
  `--max-steps` overrides both.
- `TRAIN_LEARNING_RATE`: AdamW learning rate; `--lr` overrides it.
- `TRAIN_WARMUP_STEPS`: linear warmup optimizer steps.
- `TRAIN_TARGET_SECONDS`: optional wall-clock deadline for decoder long runs.
- `TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS`: checkpoint cadence;
  `--checkpoint-interval` overrides it.
- `TRAIN_SEED`: overrides the native config seed.
- `TRAIN_PACKED_CACHE_DIR`: packed-cache directory, default
  `${DATA_DIR}/datasets/packed/train-causal_lm_full-seq1024`;
  `--packed-cache` overrides it.

Unsupported knobs must be removed from `TRAIN_CONFIG`; the native loader rejects
unknown training-config keys instead of silently ignoring them.

## Required Artifacts

- `data/train/checkpoints/latest/manifest.json`
- `data/train/checkpoints/latest/weights.index.json`
- `data/train/checkpoints/latest/weights.lkjw`
- `data/train/checkpoints/latest/trainer_state.json`
- `data/train/checkpoints/latest/optimizer.index.json`
- `data/train/checkpoints/latest/optimizer.lkjw`
- `data/train/checkpoints/final/manifest.json`
- `data/train/exports/${MODEL_NAME}/manifest.json`
- `data/models/${MODEL_NAME}/manifest.json`
- `data/train/runs/train-report.json`

## RTX 3070 Presets

- `native_debug_bf16.json`: routine debug shape.
- `decoder_40m_bf16_3070.json`: accepted decoder 40M seq1024 target.
- `native_dense_40m_bf16_3070.json`: dense 40M historical profile.
- `native_40m_bf16.json`: legacy scratch 40M shape for bounded manual runs.

## Decoder 40M Defaults

`configs/training/decoder_2h_40m_3070.json` is the source of truth:

- sequence length `1024`
- batch size `1`
- gradient accumulation `16`
- learning rate `0.0003`
- warmup `128` optimizer steps
- optimizer-step cap `1000000`
- wall-clock target `7200` seconds
- latest-checkpoint cadence `512` optimizer steps

## Accounting

Training writes `runs/train-report.json` and prints compact JSON with the same
schema. Reports include microsteps, optimizer steps, gradient accumulation, all
input tokens seen, loss-bearing tokens seen, tokens/sec, dtype/precision mode,
CUDA capability, config and dataset digests, artifact checksums, and BF16 export
logits-check status.

## Checkpoints

Training writes dense checkpoint/export directories. Public checkpoint paths are:

- `checkpoints/latest/`: newest complete resumable training state.
- `checkpoints/final/`: final training checkpoint.

`latest/` includes BF16 checkpoint weights, FP32 master weights, FP32 Adam
moments, optimizer step, microsteps, batch size, sequence length, gradient
accumulation, loss, and checksum. Resume rejects incompatible manifest/config,
model shape, vocab, seed, batch, sequence length, gradient accumulation, or
dense tensor shape. The trainer also writes `checkpoints/best/`, uses staged
latest/final writes, and records scheduler and stop-reason fields.

## Start Check

Use this bounded command to verify that Docker can start the 40M path without
running the full long job:

```bash
docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/train-start-check-decoder \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_CONFIG=/workspace/configs/training/decoder_2h_40m_3070.json \
  -e TRAIN_NATIVE_CONFIG=/workspace/configs/native/decoder_40m_bf16_3070.json \
  -e TRAIN_PACKED_CACHE_DIR=/app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  -e TRAIN_TOKENIZER=/app/data/train/tokenizer/tokenizer.json \
  -e TRAIN_DECODER_PARITY_MODE=sampled \
  -e TRAIN_DECODER_PARITY_FIRST_STEPS=1 \
  train --train --mode decoder
```

## Deadline Run

Wall-clock deadline stopping is implemented for decoder native runs. The
accepted config uses a two-hour deadline plus optimizer-step cap:

```bash
docker compose --profile train run -d \
  --name lkjai-decoder-40m-accepted-20260519 \
  -e DATA_DIR=/app/data/decoder-40m-accepted-20260519 \
  -e TRAIN_CONFIG=/workspace/configs/training/decoder_2h_40m_3070.json \
  -e TRAIN_NATIVE_CONFIG=/workspace/configs/native/decoder_40m_bf16_3070.json \
  -e TRAIN_TOKENIZER=/app/data/train/tokenizer/tokenizer.json \
  -e TRAIN_DECODER_PARITY_MODE=sampled \
  -e TRAIN_DECODER_PARITY_INTERVAL=128 \
  -e TRAIN_DECODER_PARITY_FIRST_STEPS=1 \
  train
```

Monitor with `docker logs -f lkjai-decoder-40m-accepted-20260519`.

## Two-Hour Decoder BF16 Run

Use the train Compose profile for reproducible RTX 3070 two-hour jobs. Build
and validate the seq1024 cache first, then start the default decoder command:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer
docker compose --profile corpus run --rm corpus validate-public-pretrain
docker compose --profile corpus run --rm corpus build-public-pretrain-cache
docker compose --profile train up --build train
```

The runner stores raw outputs under
`artifacts/benchmarks/<run-id>/decoder_2h_bf16_cuda/repeat-01/` and training
artifacts under `data/perf-runs/<run-id>/decoder_2h_bf16_cuda/`.

## Four-Hour Chat Attempt

This lane is not acceptance. It intentionally exports over
`data/models/decoder-40m-3070` so the browser can attempt chat through the same
serving name after SFT-style training.

Build the tokenizer and assistant-masked SFT cache:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer

docker compose --profile corpus run --build --rm corpus lkjai-native-packed-cache build \
  --source /app/data/corpus/generated/kimi-sft-60m/train \
  --tokenizer /app/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/decoder_40m_bf16_3070.json \
  --out /app/data/train/datasets/packed/train-assistant_masked_sft-seq128 \
  --split train \
  --objective assistant_masked_sft \
  --seq-len 128 \
  --run-id decoder-40m-chat-attempt-4h
```

Run a one-step start check before spending the full four hours:

```bash
TRAIN_CONFIG=/workspace/configs/training/decoder_4h_chat_attempt_3070.json \
TRAIN_MODEL_NAME=decoder-40m-3070 \
TRAIN_RUN_PURPOSE=chat_attempt \
TRAIN_DATA_DIR=/app/data/chat-attempt-start-check/train \
TRAIN_MAX_OPTIMIZER_STEPS=1 \
TRAIN_TARGET_SECONDS=0 \
docker compose --profile train run --build --rm train --train --mode decoder
```

Then run the deadline-bounded training job:

```bash
TRAIN_CONFIG=/workspace/configs/training/decoder_4h_chat_attempt_3070.json \
TRAIN_MODEL_NAME=decoder-40m-3070 \
TRAIN_RUN_PURPOSE=chat_attempt \
docker compose --profile train up --build train
```

After training, start the chat path and open the browser:

```bash
MODEL_NAME=decoder-40m-3070 docker compose --profile sandbox up --build -d
docker compose --profile web up --build -d web
curl --fail http://127.0.0.1:8081/v1/models
curl --fail http://127.0.0.1:8082/healthz
curl --fail http://127.0.0.1:8082/api/model
```

Use `http://127.0.0.1:8080`. The expected result is not quality acceptance; it
is that `/api/chat` reaches the real model and the page shows assistant content
or a concrete failure `stop_reason`. Expected disclosure remains
`lkjai_decode_accepted=false`.
