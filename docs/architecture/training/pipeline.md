# Scratch Training Pipeline

## Goal

Train, export, and evaluate the active 40M scratch model using XML actions and
the same real tool loop that production will use.

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

1. Validate tagged JSON source files in `corpus/sources/`.
2. Build fixtures and the mainline 60K corpus.
3. Build the full 500M-token public pretraining corpus under
   `data/public-corpus/` in validated JSONL shards.
4. Deduplicate and emit `train`, `val`, and `holdout` split files.
5. Train the tokenizer on the train split only.
6. Validate schema, split metadata, and write `validation-report.json`.
7. Train the scratch model from a disk-backed packed token cache. Real
   non-quick runs use the mapped dataloader by default.
8. Measure validation loss periodically on the validation split.
9. Save atomic checkpoints: `latest/` during training, retained numbered
   `steps/` snapshots, `best/` on validation improvement, and `final/` as the
   last state.
10. Run fixed eval.
11. Run raw holdout behavioral eval.
12. Record pass-rate, invalid-XML, wrong-tool, and non-finish trends.

## Defaults

- `TRAIN_PRESET=agent`
- `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json`
- `TRAIN_MODEL_PRESET=scratch-40m`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_SEQUENCE_LEN=1024`
- `TRAIN_CORPUS_TOKENS=500000000`
- `TRAIN_CORPUS_DIR=/app/data/public-corpus`
- `TRAIN_MAX_STEPS=400000` optimizer steps
- `TRAIN_BATCH_SIZE=2`
- `TRAIN_GRADIENT_ACCUMULATION=4`
- `TRAIN_BATCH_POLICY=oom_fallback`
- `TRAIN_AUTO_BATCH=true`
- `TRAIN_TARGET_EFFECTIVE_BATCH_TOKENS` defaults to batch x sequence x
  accumulation and is preserved when CUDA auto-batch adjusts microbatch size.
- `TRAIN_LR_SCHEDULE=linear_warmup_cosine`
- `TRAIN_WARMUP_STEPS=min(100, TRAIN_MAX_STEPS / 10)`
- `TRAIN_LR_MIN_FACTOR=0.1`
- `TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS=3000`
- `TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS=3000`
- `TRAIN_INTERMEDIATE_SAVE_EVERY_OPTIMIZER_STEPS=120000`
- `TRAIN_KEEP_LAST_CHECKPOINTS=8`
- `TRAIN_CHECKPOINT_RESUME_SOURCE=latest`
- `TRAIN_DATALOADER_IMPL=mapped` for real non-quick runs
- `TRAIN_STATIC_SHAPES=true`
- `TRAIN_COMPILE=auto`
- `TRAIN_COMPILE_WARMUP_MICROSTEPS=2`
- `TRAIN_ACTIVATION_CHECKPOINT=off`
- `TRAIN_CHECKPOINT_PRESERVE_RNG=false`
- `TRAIN_ATTENTION_BACKEND=auto`
- `TRAIN_CURRICULUM=configs/curriculum/agent_40m.toml`
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
- On CUDA, automatic batch sizing probes the largest safe microbatch up to
  `TRAIN_AUTO_BATCH_MAX`, then recomputes gradient accumulation from
  `TRAIN_TARGET_EFFECTIVE_BATCH_TOKENS`.
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
- Active full corpus: `data/public-corpus`
- Tokenizer: `data/train/tokenizer`
- Checkpoints: `data/train/checkpoints`
- Exports: `data/train/exports`
- Eval reports: `data/train/runs`

## Checkpoint Resume

Checkpoints are snapshot-based and atomically promoted after the complete model
and training state are written. The checkpoint manifest records
`latest_checkpoint_dir`, retained intermediate checkpoints, `best_checkpoint_dir`,
and `final_checkpoint_dir`.

`TRAIN_RESUME=auto` loads `TRAIN_CHECKPOINT_RESUME_SOURCE=latest` by default.
The training state includes optimizer, scheduler, scaler, RNG, counters, best
metric, and validation history. Exact sampler-position resume is not tracked;
after resume the dataloader may restart at a loader boundary while optimizer
and scheduler state remain exact.
