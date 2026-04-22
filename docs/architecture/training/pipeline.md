# Training Pipeline

## Commands

- `docker compose --profile train up --build train`.
- `python -m lkjai_train.cli prepare-fixtures`.
- `python -m lkjai_train.cli validate-dataset`.
- `python -m lkjai_train.cli train-adapter`.
- `python -m lkjai_train.cli fixed-eval`.
- `python -m lkjai_train.cli export-manifest`.
- `python -m lkjai_train.cli smoke`.

## Pipeline Order

1. Prepare built-in starter dataset or load external JSONL dataset.
2. Validate message/tool trajectory schema and split metadata.
3. Run QLoRA adapter training with Transformers + PEFT + bitsandbytes.
4. Persist adapter checkpoints and train metadata.
5. Run fixed evals and write report under `data/train/runs/`.
6. Export manifest metadata for serving handoff.
7. Keep deterministic `smoke` command for CI-only fast checks.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_PRESET=agent` is the Compose default.
- `TRAIN_PRESET=quick` is reserved for deterministic smoke checks.
- `TRAIN_PRESET=custom` reads explicit `TRAIN_*` knobs.
- `TRAIN_DATA_DIR` defaults to `/app/data/train`.
- `TRAIN_BASE_MODEL` defaults to `Qwen/Qwen3-0.6B`.
- `TRAIN_SEQUENCE_LEN=2048`.
- `TRAIN_LORA_RANK=16`.
- `TRAIN_LOAD_IN_4BIT=1`.

## Artifacts

- Datasets live under `data/train/datasets`.
- Adapters live under `data/train/adapters`.
- Exports live under `data/train/exports`.
- Eval reports live under `data/train/runs`.
