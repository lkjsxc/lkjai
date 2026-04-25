# Scratch Training Run Contract

## Goal

Run one measurable long training job for the 3070-first scratch model.

## Default Behavior

- `docker compose --profile train up --build train` runs preset-driven training.
- `TRAIN_PRESET=agent` is the RTX 3070 long-run target.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.
- Training defaults to `TRAIN_OBJECTIVE=causal_lm_full`.
- Export defaults to the best validation checkpoint.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, `custom`, `scratch-30m-debug`,
  `scratch-60m`, `scratch-93m-max`
- `TRAIN_MODEL_PRESET`: default `scratch-60m`
- `TRAIN_OBJECTIVE`: `causal_lm_full` or `assistant_masked_sft`
- `TRAIN_VOCAB_SIZE`: default `8192`
- `TRAIN_SEQUENCE_LEN`: default `1024`
- `TRAIN_LEARNING_RATE`: default `3e-4`
- `TRAIN_DROPOUT`: default `0.0`
- `TRAIN_BATCH_SIZE`: default `1`
- `TRAIN_GRADIENT_ACCUMULATION`: default `8`
- `TRAIN_MAX_STEPS`: optimizer steps, default `120000`
- `TRAIN_MAX_OPTIMIZER_STEPS`: explicit optimizer-step cap
- `TRAIN_MAX_MICROSTEPS`: optional microstep hard cap, `0` disables it
- `TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS`: default `250`
- `TRAIN_SAVE_EVERY_OPTIMIZER_STEPS`: default `1000`
- `TRAIN_VALIDATION_BATCHES`: default `8`
- `TRAIN_RESUME`: `auto`, `never`, or `required`
- `TRAIN_AMP`: `auto`, `bf16`, or `fp16`
- `TRAIN_TORCH_COMPILE`: optional PyTorch compile flag, default disabled
- `TRAIN_EXPORT_CHECKPOINT`: `best` or `final`, default `best`
- `TRAIN_CORPUS_SIZE`: default `60000`
- `TRAIN_FIXED_EVAL_THRESHOLD`: default `0.60` for fixed report metadata
- `TRAIN_BEHAVIORAL_THRESHOLD`: default `0.35` until the next ladder is passed
- `TRAIN_ENFORCE_COMPETENCY`: fail command when behavioral gates are missed

## Required Artifacts

- `data/train/datasets/metadata.json`
- `data/train/tokenizer/manifest.json`
- `data/train/checkpoints/manifest.json`
- `data/train/exports/manifest.json`
- `data/train/runs/fixed-eval.json`
- `data/train/runs/behavioral-eval.json`

## RTX 3070 Presets

- `quick`: tiny smoke/debug run.
- `scratch-30m-debug`: about 29.9M parameters, context 1024.
- `scratch-60m`: about 58.8M parameters, context 1024.
- `scratch-93m-max`: about 93.3M parameters, context 1024, intended for
  microbatch 1 with activation checkpointing.

## Accounting

Training summaries report microsteps, optimizer steps, gradient accumulation,
all input tokens seen, loss-bearing tokens seen, tokens/sec,
loss-bearing tokens/sec, and effective batch tokens per optimizer step.
