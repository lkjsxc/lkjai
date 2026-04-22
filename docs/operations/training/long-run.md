# Agent Tuning Run Contract

## Default Behavior

- `docker compose --profile train up --build train` starts quick validation by default.
- `TRAIN_PRESET=agent` starts RTX 3070-oriented QLoRA tuning.
- The train service writes under `TRAIN_DATA_DIR`, default `/app/data/train`.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, or `custom`.
- `TRAIN_BASE_MODEL`: default `Qwen/Qwen3-0.6B`.
- `TRAIN_SEQUENCE_LEN`: default `2048`.
- `TRAIN_LORA_RANK`: default `16`.
- `TRAIN_LOAD_IN_4BIT`: default enabled.
- `TRAIN_ENFORCE_COMPETENCY`: fail train command when eval is below threshold.

## Stop Rule

- Quick preset stops after fixture validation and fixed eval.
- Agent preset is controlled by the tuning backend configuration.
- Failed dataset validation stops before any GPU training.

## Artifacts

- Dataset fixtures: `data/train/datasets/fixtures.jsonl`
- Adapter output: `data/train/adapters/`
- Export manifest: `data/train/exports/manifest.json`
- Fixed eval summary: `data/train/runs/fixed-eval.json`

## Compose Examples

```bash
docker compose --profile train up --build train
TRAIN_PRESET=agent docker compose --profile train up --build train
TRAIN_PRESET=custom TRAIN_BASE_MODEL=Qwen/Qwen3-0.6B docker compose --profile train up --build train
```

## Non-Goal

- The verification profile does not run GPU QLoRA tuning.
