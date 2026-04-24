# Scratch Training Run Contract

## Goal

Run one measurable long training job for the 3070-first scratch model.

## Default Behavior

- `docker compose --profile train up --build train` runs preset-driven training.
- `TRAIN_PRESET=agent` is the RTX 3070 long-run target.
- Training writes under `TRAIN_DATA_DIR`, default `/app/data/train`.

## Required Environment Knobs

- `TRAIN_PRESET`: `quick`, `agent`, `custom`
- `TRAIN_MODEL_PRESET`: default `scratch-60m`
- `TRAIN_VOCAB_SIZE`: default `8192`
- `TRAIN_SEQUENCE_LEN`: default `1024`
- `TRAIN_LEARNING_RATE`: default `3e-4`
- `TRAIN_BATCH_SIZE`: default `1`
- `TRAIN_GRADIENT_ACCUMULATION`: default `8`
- `TRAIN_MAX_STEPS`: default `3000`
- `TRAIN_CORPUS_SIZE`: default `12000`
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
