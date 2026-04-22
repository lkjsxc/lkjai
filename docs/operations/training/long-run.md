# Agent Training Run Contract

## Goal

Run real adapter training and produce measurable artifacts for local serving and
evaluation.

## Default Behavior

- `docker compose --profile train up --build train` runs preset-driven training.
- `TRAIN_PRESET=agent` is the RTX 3070 long-run target.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, `custom`.
- `TRAIN_BASE_MODEL`: default `Qwen/Qwen3-0.6B`.
- `TRAIN_SEQUENCE_LEN`: default `2048`.
- `TRAIN_LORA_RANK`: default `16`.
- `TRAIN_LORA_ALPHA`: default `32`.
- `TRAIN_LEARNING_RATE`: default `1e-4`.
- `TRAIN_LOAD_IN_4BIT`: default enabled.
- `TRAIN_GRADIENT_CHECKPOINTING`: default enabled.
- `TRAIN_EPOCHS`: default `1`.
- `TRAIN_BATCH_SIZE`: default `1`.
- `TRAIN_GRADIENT_ACCUMULATION`: default `8`.
- `TRAIN_MAX_STEPS`: default `200`.
- `TRAIN_EVAL_RATIO`: default `0.2`.
- `TRAIN_FIXED_EVAL_THRESHOLD`: default `0.80`.
- `TRAIN_ENFORCE_COMPETENCY`: fail command when eval is below threshold.

## Stop Rules

- Dataset validation failure stops before GPU training.
- Training errors stop pipeline with non-zero exit.
- Competency enforcement stops acceptance when below threshold.

## Required Artifacts

- Dataset metadata: `data/train/datasets/metadata.json`
- Adapter checkpoints: `data/train/adapters/`
- Export manifest: `data/train/exports/manifest.json`
- Fixed eval summary: `data/train/runs/fixed-eval.json`

## Compose Examples

```bash
docker compose --profile train up --build train
TRAIN_PRESET=agent docker compose --profile train up --build train
TRAIN_PRESET=custom TRAIN_BASE_MODEL=Qwen/Qwen3-0.6B docker compose --profile train up --build train
```

## Boundary

- Verify profile remains deterministic and lightweight.
- Long-run quality decisions depend on training artifacts and eval reports, not
  verify-only smoke output.
