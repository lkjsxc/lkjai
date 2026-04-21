# Long-Run Training Contract

## Default Behavior

- `docker compose --profile train up --build` starts long-run training by default.
- Default target runtime is `TRAIN_MAX_DURATION_SECS=21600` (~6 hours).
- The train service writes under `TRAIN_DATA_DIR`, default `/app/data/train`.

## Required Environment Knobs

- `TRAIN_PRESET`: training profile selector. Default Compose value is `longrun`.
- `TRAIN_MAX_DURATION_SECS`: wall-clock budget in seconds.
- `TRAIN_STEPS`: minimum optimization steps before duration stop is allowed.
- `TRAIN_TOKEN_BUDGET`: corpus token budget.
- `TRAIN_DATASET`: dataset source, default `HuggingFaceFW/fineweb-edu`.
- `TRAIN_CONFIG`: model config path.
- `TRAIN_CONTEXT`: optional context override (`0` keeps config value).

## Stop Rule

- Training must run until both conditions are true:
  1. Minimum `TRAIN_STEPS` has completed.
  2. `TRAIN_MAX_DURATION_SECS` wall-clock budget has elapsed.
- If `TRAIN_MAX_DURATION_SECS` is `0`, stop is step-driven only.

## Artifacts

- Checkpoint: `data/train/checkpoints/latest.pt`
- Training summary: `data/train/runs/last-train.json`
- Export: `data/train/models/lkj-150m/{model.safetensors,config.json,tokenizer.json,size.json}`
- Fixed eval summary: `data/train/runs/fixed-eval.json`

## Compose Examples

```bash
docker compose --profile train up --build
TRAIN_MAX_DURATION_SECS=7200 TRAIN_STEPS=256 docker compose --profile train up --build
TRAIN_PRESET=quick docker compose --profile train up --build
```

## Non-Goal

- The verification profile is not a six-hour run. Verification stays smoke-scale.
