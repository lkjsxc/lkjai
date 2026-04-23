# Scratch Training Run Contract

## Goal

Run scratch tokenizer and language-model training with measurable artifacts for
local serving and behavioral evaluation.

## Default Behavior

- `docker compose --profile train up --build train` runs preset-driven training.
- `TRAIN_PRESET=agent` is the RTX 3070 long-run target.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, `custom`.
- `TRAIN_MODEL_PRESET`: default `scratch-60m`.
- `TRAIN_VOCAB_SIZE`: default `8192`.
- `TRAIN_SEQUENCE_LEN`: default `1024`.
- `TRAIN_LAYERS`: default preset-owned.
- `TRAIN_HIDDEN_SIZE`: default preset-owned.
- `TRAIN_HEADS`: default preset-owned.
- `TRAIN_KV_HEADS`: default preset-owned.
- `TRAIN_LEARNING_RATE`: default `3e-4`.
- `TRAIN_GRADIENT_CHECKPOINTING`: default enabled.
- `TRAIN_BATCH_SIZE`: default `1`.
- `TRAIN_GRADIENT_ACCUMULATION`: default `8`.
- `TRAIN_MAX_STEPS`: default `3000`.
- `TRAIN_CORPUS_SIZE`: default `4000`.
- `TRAIN_FIXED_EVAL_THRESHOLD`: default `0.80`.
- `TRAIN_ENFORCE_COMPETENCY`: fail command when eval is below threshold.

## Stop Rules

- Dataset validation failure stops before model training.
- Tokenizer training failure stops before checkpoint training.
- Training errors stop pipeline with non-zero exit.
- Competency enforcement stops acceptance when below threshold.
- Behavioral eval failure triggers data/config iteration before acceptance.

## Required Artifacts

- Dataset metadata: `data/train/datasets/metadata.json`
- Tokenizer manifest: `data/train/tokenizer/manifest.json`
- Checkpoint manifest: `data/train/checkpoints/manifest.json`
- Export manifest: `data/train/exports/manifest.json`
- Fixed eval summary: `data/train/runs/fixed-eval.json`
- Behavioral eval summary: `data/train/runs/behavioral-eval.json`
- Preference pairs: `data/train/preferences/pairs.jsonl`
- DPO summary: `data/train/checkpoints/dpo-summary.json`

## Compose Examples

```bash
docker compose --profile train up --build train
TRAIN_PRESET=quick docker compose --profile train up --build train
TRAIN_PRESET=custom TRAIN_MAX_STEPS=50 docker compose --profile train up --build train
```

## Boundary

- Verify profile remains deterministic and lightweight.
- Long-run quality decisions depend on behavioral eval and saved inference
  transcripts, not artifact existence alone.
