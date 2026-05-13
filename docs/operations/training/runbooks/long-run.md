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
- `TRAIN_MODEL_NAME=dense-40m-3070` is the default training export name
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
  `MODEL_NAME` and defaults to `dense-40m-3070`.
- `TRAIN_SEQUENCE_LEN`: sequence length; `--seq-len` overrides it.
- `TRAIN_BATCH_SIZE`: microbatch size; `--batch-size` overrides it.
- `TRAIN_GRADIENT_ACCUMULATION`: microsteps per AdamW step; `--grad-accum`
  overrides it.
- `TRAIN_MAX_STEPS` or `TRAIN_MAX_OPTIMIZER_STEPS`: optimizer-step cap;
  `--max-steps` overrides both.
- `TRAIN_LEARNING_RATE`: AdamW learning rate; `--lr` overrides it.
- `TRAIN_WARMUP_STEPS`: linear warmup optimizer steps.
- `TRAIN_TARGET_SECONDS`: optional wall-clock deadline for dense long runs.
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
- `native_dense_20m_bf16_3070.json`: explicit dense 20M-size seq1024 shape
  for the two-hour BF16 runner.
- `native_dense_40m_bf16_3070.json`: accepted dense 40M-size seq1024 shape for
  browser-demo long runs.
- `native_40m_bf16.json`: legacy scratch 40M shape for bounded manual runs.

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
  -e DATA_DIR=/app/data/train-start-check \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_NATIVE_CONFIG=/workspace/configs/native/native_dense_40m_bf16_3070.json \
  train --train
```

## Deadline Run

Wall-clock deadline stopping is implemented for dense and decoder native runs.
The accepted dense config uses a two-hour deadline plus optimizer-step cap:

```bash
docker compose --profile train run -d \
  --name lkjai-dense-40m-accepted-20260512 \
  -e DATA_DIR=/app/data/dense-40m-accepted-20260512 \
  -e TRAIN_CONFIG=/workspace/configs/training/dense_40m_accepted_3070.json \
  train
```

Monitor with `docker logs -f lkjai-dense-40m-accepted-20260512`.

## Two-Hour Dense BF16 Run

Use the train Compose profile for reproducible RTX 3070 two-hour jobs. Build
and validate the seq1024 cache first, then start the default dense 40M command:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer
docker compose --profile corpus run --rm corpus validate-public-pretrain
docker compose --profile corpus run --rm corpus build-public-pretrain-cache
docker compose --profile train up --build train
```

The runner stores raw outputs under
`artifacts/benchmarks/<run-id>/dense_2h_bf16_cuda/repeat-01/` and training
artifacts under `data/perf-runs/<run-id>/dense_2h_bf16_cuda/`.
