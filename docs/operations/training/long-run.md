# Scratch Training Run Contract

## Goal

Run one measurable long training job for the 3070-first 40M scratch model.

## Default Behavior

- `docker compose --profile train up --build train` reads the committed JSON
  training config and starts training.
- `TRAIN_PRESET=agent` is the RTX 3070 long-run target.
- `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json` is the
  default long-run config.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.
- Training defaults to `TRAIN_OBJECTIVE=causal_lm_full`.
- Export defaults to the best validation checkpoint.
- Real non-quick runs default to the batch-mapped packed-token dataloader.
- CUDA runs probe a safe microbatch automatically and adjust gradient
  accumulation to keep the target effective token batch.
- Resume defaults to the newest complete `latest/` checkpoint snapshot.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, `custom`, `scratch-20m`,
  `scratch-30m-debug`, `scratch-40m`, `scratch-60m`, `scratch-93m-max`
- `TRAIN_CONFIG`: JSON config path, default 40M 12-hour estimate in Compose
- `TRAIN_MODEL_PRESET`: default `scratch-40m`
- `TRAIN_OBJECTIVE`: `causal_lm_full` or `assistant_masked_sft`
- `TRAIN_VOCAB_SIZE`: default `8192`
- `TRAIN_SEQUENCE_LEN`: default `1024`
- `TRAIN_LEARNING_RATE`: default `3e-4`
- `TRAIN_WEIGHT_DECAY`: default `0.01`
- `TRAIN_BETA1`: AdamW beta1, default `0.9`
- `TRAIN_BETA2`: AdamW beta2, default `0.999`
- `TRAIN_EPS`: AdamW epsilon, default `1e-8`
- `TRAIN_LR_SCHEDULE`: `linear_warmup_cosine`, `cosine`, or `constant`
- `TRAIN_WARMUP_STEPS`: default `min(100, TRAIN_MAX_STEPS / 10)`
- `TRAIN_LR_MIN_FACTOR`: cosine floor factor, default `0.1`
- `TRAIN_DROPOUT`: default `0.0`
- `TRAIN_BATCH_SIZE`: default `2`
- `TRAIN_GRADIENT_ACCUMULATION`: default `4`
- `TRAIN_BATCH_POLICY`: `fixed`, `oom_fallback`, or timed CUDA `sweep`;
  default `oom_fallback`
- `TRAIN_AUTO_BATCH`: default `true`; CUDA-only automatic microbatch probe
- `TRAIN_AUTO_BATCH_MAX`: default `16`
- `TRAIN_TARGET_EFFECTIVE_BATCH_TOKENS`: default batch x sequence x accumulation
- `TRAIN_MAX_STEPS`: optimizer steps, default `400000`
- `TRAIN_MAX_OPTIMIZER_STEPS`: explicit optimizer-step cap
- `TRAIN_MAX_MICROSTEPS`: optional microstep hard cap, `0` disables it
- `TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS`: default `3000`
- `TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS`: default `3000` for non-quick,
  `1` for quick
- `TRAIN_INTERMEDIATE_SAVE_EVERY_OPTIMIZER_STEPS`: default `120000` for
  non-quick, `0` for quick
- `TRAIN_KEEP_LAST_CHECKPOINTS`: default `8`
- `TRAIN_CHECKPOINT_RESUME_SOURCE`: `latest`, `final`, or `best`; default
  `latest`
- `TRAIN_LOG_EVERY_OPTIMIZER_STEPS`: default `250`
- `TRAIN_SAVE_EVERY_OPTIMIZER_STEPS`: legacy compatibility knob
- `TRAIN_VALIDATION_BATCHES`: default `8`
- `TRAIN_RESUME`: `auto`, `never`, or `required`
- `TRAIN_AMP`: `auto`, `bf16`, or `fp16`
- `TRAIN_COMPILE`: `off`, `auto`, `default`, `reduce-overhead`,
  `max-autotune`, or `max-autotune-no-cudagraphs`; default `auto`
- `TRAIN_COMPILE_WARMUP_MICROSTEPS`: default `2`
- `TRAIN_STATIC_SHAPES`: default `true`
- `TRAIN_ACTIVATION_CHECKPOINT`: `off`, `all`, or `every_n`; default `off`
- `TRAIN_ACTIVATION_CHECKPOINT_EVERY_N`: default `2`
- `TRAIN_CHECKPOINT_PRESERVE_RNG`: default `false`
- `TRAIN_ATTENTION_BACKEND`: `auto`, `sdpa`, `sdpa_flash`, `sdpa_math`, or
  `flash2`; default `auto`
- `TRAIN_EXPORT_CHECKPOINT`: `best` or `final`, default `best`
- `TRAIN_DATALOADER_IMPL`: `batch_mapped` by default for real non-quick runs,
  `legacy` for quick; `mapped` remains a benchmark comparison path
- `TRAIN_DATALOADER_BENCHMARK`: optional benchmark logging flag
- `TRAIN_CORPUS_SIZE`: default `120000`
- `TRAIN_FIXED_EVAL_THRESHOLD`: default `0.60` for fixed report metadata
- `TRAIN_BEHAVIORAL_THRESHOLD`: default `0.35` until the next ladder is passed
- `TRAIN_ENFORCE_COMPETENCY`: fail command when behavioral gates are missed

## Required Artifacts

- `data/train/datasets/metadata.json`
- `data/train/tokenizer/manifest.json`
- `data/train/checkpoints/manifest.json`
- `data/train/checkpoints/latest/model.pt`
- `data/train/exports/manifest.json`
- `data/train/runs/fixed-eval.json`
- `data/train/runs/behavioral-eval.json`

## RTX 3070 Presets

- `quick`: tiny smoke/debug run.
- `scratch-20m`: about 20.1M parameters, context 1024.
- `scratch-30m-debug`: about 29.9M parameters, context 1024.
- `scratch-40m`: about 39.6M parameters, context 1024.
- `scratch-60m`: about 58.8M parameters, context 1024.
- `scratch-93m-max`: about 93.3M parameters, context 1024, intended for
  microbatch 1 with activation checkpointing.

## Accounting

Training summaries report microsteps, optimizer steps, gradient accumulation,
all input tokens seen, loss-bearing tokens seen, tokens/sec,
loss-bearing tokens/sec, and effective batch tokens per optimizer step.

## Checkpoints

Training writes complete state checkpoints atomically through temporary paths
on the same filesystem before promotion. Public checkpoint paths remain:

- `checkpoints/latest/`: newest complete resumable training state.
- `checkpoints/best/`: best validation checkpoint.
- `checkpoints/final/`: final training checkpoint.
- `checkpoints/steps/step-000001/`: retained numbered intermediate snapshots.

`latest/` is preferred for `TRAIN_RESUME=auto` and includes model weights,
optimizer, scheduler, scaler, RNG, counters, best metric, validation history,
and settings. Temporary incomplete checkpoint directories are ignored. Data
order can restart at a loader boundary after resume; optimizer, scheduler, RNG,
and counters resume from the checkpoint state.

## Start Check

Use this bounded command to verify that Docker can start the 40M path without
running the full long job:

```bash
docker compose --profile train run --rm \
  -e TRAIN_DATA_DIR=/app/data/train-start-check \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_MAX_STEPS=1 \
  -e TRAIN_RESUME=never \
  -e TRAIN_COMPILE=off \
  train train-scratch --preset agent
```
