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

1. Prepare fixture or external JSONL datasets.
2. Validate message and tool trajectory schema.
3. Run QLoRA tuning for the configured dense model.
4. Run fixed agent evals.
5. Merge adapter when requested.
6. Convert and quantize to GGUF when requested.
7. Write manifest metadata beside artifacts.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_PRESET=quick` is the Compose default.
- `TRAIN_PRESET=agent` runs RTX 3070-oriented QLoRA settings.
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
