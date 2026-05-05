# Scratch Training Pipeline

## Goal

Train and export the current dense BF16 CUDA foundation while preserving the
40M XML-action pipeline as the target operating path.

## Commands

- `docker compose --profile train up --build train`
- `docker compose --profile verify up --build --abort-on-container-exit verify`
- `lkjai-native-train --smoke --steps 2`
- `lkjai-native-inspect --model-dir data/models/lkjai-scratch-40m`

## Pipeline Order

1. Validate tagged JSON source files in `corpus/sources/`.
2. Read reviewed JSONL corpus rows.
3. Serialize dialogue and assistant action targets.
4. Train or load the byte-level BPE tokenizer.
5. Write `lkjai-packed-cache-v2` train, val, and holdout caches.
6. Train the causal-LM dense foundation through native C++/CUDA.
7. Save native dense checkpoints and `lkjai-native-artifact-v2` exports.
8. Run artifact inspect and dense logits checks.
9. Confirm native server chat rejects dense/transformer decode and labels
   decoder partial decode honestly when a decoder artifact is present.
10. Add XML-action SFT and behavioral eval only after decode lands.

## Defaults

- `TRAIN_PRESET=agent`
- `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json`
- `TRAIN_NATIVE_CONFIG=/workspace/configs/native/native_40m_bf16.json`
- `TRAIN_SEQUENCE_LEN=1024`
- `TRAIN_MAX_STEPS=400000` optimizer steps
- `TRAIN_BATCH_SIZE=2`
- `TRAIN_GRADIENT_ACCUMULATION=4`
- `TRAIN_LEARNING_RATE=0.0003`
- `TRAIN_WARMUP_STEPS=100`
- `TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS=3000`
- Native packed-cache reader for `--train`
- `TRAIN_DATA_DIR=/app/data/train`

## Objectives And Accounting

- `causal_lm_full`: full next-token causal LM training. Every non-padding next
  token contributes to loss.
- `assistant_masked_sft`: target XML-action objective after tokenizer,
  transformer, and decode milestones.
- A microstep is one forward/backward batch.
- An optimizer step happens after `TRAIN_GRADIENT_ACCUMULATION` microsteps.
- `TRAIN_MAX_STEPS` and `TRAIN_MAX_OPTIMIZER_STEPS` stop by optimizer steps.
- Reports count input tokens and loss-bearing tokens.

Recommended stages:

1. `causal_lm_full` on curated Cosmopedia `text`-only public pretraining.
2. `assistant_masked_sft` on first-party XML action traces.
3. Optional later preference training after both objective gates pass.

No current artifact is accepted for product chat. Decoder artifacts may serve
host-reference recompute choices, but accepted autoregressive KV-cache decode is
still a native milestone.

## Artifacts

- Datasets: `data/train/datasets`
- Active full corpus: `data/public-corpus`
- Active first-party SFT corpus: `corpus/generated/kimi-sft-60m-v2` after it
  reaches `60000000` validated tokens. A promoted 1M-token v2 pilot is the
  required intermediate gate. No v1 Kimi corpus is active.
- Tokenizer: `data/train/tokenizer`
- Checkpoints: `data/train/checkpoints`
- Native exports: `data/models/lkjai-scratch-40m`
- Stable train report: `data/train/runs/train-report.json`

## Checkpoint Resume

Checkpoints contain dense model weights, FP32 master tensors, FP32 Adam moments,
optimizer step, microsteps, batch size, sequence length, gradient accumulation,
loss, and checksum. `--resume` restores the FP32 masters and Adam moments,
rebuilds BF16 CUDA shadows, and rejects incompatible config, model shape, vocab,
seed, batch, sequence, gradient accumulation, or dense tensor shape. Scheduler,
scaler, retained intermediate checkpoints, and best checkpoint selection are
target additions.
