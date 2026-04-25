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
2. Build fixtures and the mainline 60K corpus.
3. Build the full 500M-token Kimi corpus under
   `training/corpus/kimi-synthetic-v1/` in validated JSONL shards.
4. Deduplicate and emit `train`, `val`, and `holdout` split files.
5. Train the tokenizer on the train split only.
6. Validate schema, split metadata, and write `validation-report.json`.
7. Train the scratch model from a disk-backed packed token cache.
8. Measure validation loss periodically on the validation split.
9. Save `best/` on validation improvement and `final/` as the last state.
10. Run fixed eval.
11. Run raw holdout behavioral eval.
12. Record pass-rate, invalid-XML, wrong-tool, and non-finish trends.

## Defaults

- `TRAIN_PRESET=agent`
- `TRAIN_MODEL_PRESET=scratch-60m`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_SEQUENCE_LEN=1024`
- `TRAIN_CORPUS_TOKENS=500000000`
- `TRAIN_CORPUS_DIR=/app/data/kimi-corpus`
- `TRAIN_MAX_STEPS=120000` optimizer steps
- `TRAIN_GRADIENT_ACCUMULATION=8`
- `TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS=250`
- `TRAIN_SAVE_EVERY_OPTIMIZER_STEPS=1000`
- `TRAIN_EXPORT_CHECKPOINT=best`
- `TRAIN_BEHAVIORAL_THRESHOLD=0.35`
- `TRAIN_DATA_DIR=/app/data/train`

## Objectives And Accounting

- `causal_lm_full`: full next-token causal LM training. Every non-padding next
  token contributes to loss.
- `assistant_masked_sft`: message rows keep the XML serialization path, but
  only assistant content tokens contribute to loss.
- A microstep is one forward/backward batch.
- An optimizer step happens after `TRAIN_GRADIENT_ACCUMULATION` microsteps.
- `TRAIN_MAX_STEPS` and `TRAIN_MAX_OPTIMIZER_STEPS` stop by optimizer steps.
- `TRAIN_MAX_MICROSTEPS` is an optional hard stop for old-style microstep caps.
- `input_tokens_seen` counts all tokens fed to the model.
- `loss_tokens_seen` counts only labels that are not masked with `-100`.

Recommended stages:

1. `causal_lm_full` pretraining.
2. Optional continued pretraining on domain, tool, docs, and code mixtures.
3. `assistant_masked_sft` on XML action traces.

## Artifacts

- Datasets: `data/train/datasets`
- Committed full corpus: `training/corpus/kimi-synthetic-v1`
- Tokenizer: `data/train/tokenizer`
- Checkpoints: `data/train/checkpoints`
- Exports: `data/train/exports`
- Eval reports: `data/train/runs`
