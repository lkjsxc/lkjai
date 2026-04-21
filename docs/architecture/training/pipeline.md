# Training Pipeline

## Commands

- `docker compose --profile train up --build`
- `python -m lkjai_train.cli prepare-corpus`
- `python -m lkjai_train.cli train-tokenizer`
- `python -m lkjai_train.cli train-model`
- `python -m lkjai_train.cli export-model`
- `python -m lkjai_train.cli smoke`

## Pipeline Order

1. Stream or sample corpus rows.
2. Train tokenizer.
3. Pack token IDs into memmap shards.
4. Train model with resumable checkpoints.
5. Export fp16 safetensors, tokenizer, and config.
6. Verify artifact size.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_TINY=1` switches the same command to fixture-scale training.
- `TRAIN_TOKEN_BUDGET`, `TRAIN_DATASET`, `TRAIN_VOCAB_SIZE`, `TRAIN_STEPS`,
  `TRAIN_CONTEXT`, and `TRAIN_CONFIG` tune the default pipeline.

## Checkpoints

- Checkpoints live under `data/checkpoints`.
- Final serving exports live under `data/models`.
- Training logs live under `data/runs`.
