# Scratch Training Pipeline

## Goal

Train, export, and evaluate the 60M scratch model using XML actions and the
same real tool loop that production will use.

## Commands

- `docker compose --profile train up --build train`
- `python -m lkjai_train.cli prepare-fixtures`
- `python -m lkjai_train.cli validate-sources`
- `python -m lkjai_train.cli prepare-corpus`
- `python -m lkjai_train.cli train-tokenizer`
- `python -m lkjai_train.cli validate-dataset`
- `python -m lkjai_train.cli train-scratch`
- `python -m lkjai_train.cli fixed-eval`
- `python -m lkjai_train.cli behavioral-eval`
- `python -m lkjai_train.cli export-manifest`
- `python -m lkjai_train.cli smoke`

## Pipeline Order

1. Validate tagged JSON source files in `training/corpus_sources/`.
2. Build fixtures and the full 500M-token Kimi corpus outside git.
3. Deduplicate and emit `train`, `val`, and `holdout` split files.
4. Train the tokenizer on the train split only.
5. Validate schema and split metadata.
6. Train the scratch model on packed row boundaries from the train split.
7. Measure validation loss on the validation split.
8. Export the accepted checkpoint.
9. Run fixed eval.
10. Run raw holdout behavioral eval.
11. Record pass-rate, invalid-XML, wrong-tool, and non-finish trends.

## Defaults

- `TRAIN_PRESET=agent`
- `TRAIN_MODEL_PRESET=scratch-60m`
- `TRAIN_SEQUENCE_LEN=1024`
- `TRAIN_CORPUS_TOKENS=500000000`
- `TRAIN_CORPUS_DIR=/app/data/kimi-corpus`
- `TRAIN_MAX_STEPS=12000`
- `TRAIN_BEHAVIORAL_THRESHOLD=0.35`
- `TRAIN_DATA_DIR=/app/data/train`

## Artifacts

- Datasets: `data/train/datasets`
- Tokenizer: `data/train/tokenizer`
- Checkpoints: `data/train/checkpoints`
- Exports: `data/train/exports`
- Eval reports: `data/train/runs`
