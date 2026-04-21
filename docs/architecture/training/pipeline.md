# Training Pipeline

## Commands

- `docker compose --profile train up --build`
- `python -m lkjai_train.cli prepare-corpus`
- `python -m lkjai_train.cli train-tokenizer`
- `python -m lkjai_train.cli train-model`
- `python -m lkjai_train.cli export-model`
- `python -m lkjai_train.cli fixed-eval`
- `python -m lkjai_train.cli smoke`

## Pipeline Order

1. Stream or sample corpus rows.
2. Train tokenizer.
3. Pack token IDs into memmap shards.
4. Train model with resumable checkpoints and duration-aware stop.
5. Export fp16 safetensors, tokenizer, and config.
6. Verify artifact size.
7. Run fixed-eval and compute competency pass rate.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_PRESET=longrun` is the Compose default and targets ~6 hours.
- `TRAIN_PRESET=quick` keeps fixture-scale behavior for fast debugging.
- `TRAIN_PRESET=full` uses full corpus defaults and `lkj-150m` config.
- `TRAIN_PRESET=custom` reads each `TRAIN_*` knob directly.
- `TRAIN_DATA_DIR` controls the mounted output root and defaults to
  `/app/data/train`.
- `TRAIN_TOKEN_BUDGET`, `TRAIN_DATASET`, `TRAIN_VOCAB_SIZE`, `TRAIN_STEPS`,
  `TRAIN_MAX_DURATION_SECS`, `TRAIN_CONTEXT`, and `TRAIN_CONFIG` tune the run.
- `TRAIN_FIXED_EVAL_THRESHOLD` and `TRAIN_ENFORCE_COMPETENCY` control
  competency acceptance.

## Checkpoints

- Checkpoints live under `data/checkpoints` (default `latest.pt`).
- Final serving exports live under `data/models/lkj-150m`.
- Training logs and fixed-eval reports live under `data/runs`.
