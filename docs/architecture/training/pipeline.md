# Scratch Training Pipeline

## Goal

Train, export, and evaluate the active 40M scratch model using XML actions and
the same real tool loop that production will use.

## Commands

- `docker compose --profile train up --build train`
- `docker compose --profile verify up --build --abort-on-container-exit verify`
- `lkjai-native-train --smoke --steps 2`
- `lkjai-native-inspect --model-dir data/models/lkjai-scratch-40m`

## Pipeline Order

1. Validate tagged JSON source files in `corpus/sources/`.
2. Read reviewed JSONL corpus rows.
3. Serialize dialogue and assistant action targets.
4. Train or load the byte-level BPE tokenizer.
5. Write `lkjai-packed-cache-v2` train, val, and holdout caches.
6. Train the causal-LM pretrain stage through native C++/CUDA.
7. Train the XML-action SFT stage from accepted pretrain weights.
8. Save atomic native checkpoints and `lkjai-native-artifact-v1` exports.
9. Run native server generation checks and behavioral eval.
10. Record pass-rate, invalid-XML, wrong-tool, and non-finish trends.

## Defaults

- `TRAIN_PRESET=agent`
- `TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json`
- `TRAIN_MODEL_PRESET=scratch-40m`
- `TRAIN_OBJECTIVE=causal_lm_full`
- `TRAIN_SEQUENCE_LEN=1024`
- `TRAIN_CORPUS_TOKENS=500000000`
- `TRAIN_PUBLIC_PRETRAIN_TOKENS=500000000`
- `TRAIN_FIRST_PARTY_SFT_TOKENS=60000000`
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
- Native packed-cache reader for real non-quick runs
- `TRAIN_STATIC_SHAPES=true`
- `TRAIN_COMPILE` is not used in the native product path
- `TRAIN_ACTIVATION_CHECKPOINT=off`
- `TRAIN_CHECKPOINT_PRESERVE_RNG=false`
- `TRAIN_ATTENTION_BACKEND=auto`
- `TRAIN_CURRICULUM=configs/curriculum/agent_40m.toml`
- `TRAIN_EXPORT_CHECKPOINT=best`
- `TRAIN_BEHAVIORAL_THRESHOLD=0.35`
- `TRAIN_EVERYDAY_CHAT_THRESHOLD=0.90`
- `TRAIN_XML_VALIDITY_THRESHOLD=0.95`
- `TRAIN_DATA_DIR=/app/data/train`

## Objectives And Accounting

- `causal_lm_full`: full next-token causal LM training. Every non-padding next
  token contributes to loss.
- `assistant_masked_sft`: message rows keep the XML serialization path, but
  only assistant content tokens contribute to loss.
- A microstep is one forward/backward batch.
- An optimizer step happens after `TRAIN_GRADIENT_ACCUMULATION` microsteps.
- On CUDA, native auto-batch probes the largest safe microbatch up to
  `TRAIN_AUTO_BATCH_MAX`.
- `TRAIN_MAX_STEPS` and `TRAIN_MAX_OPTIMIZER_STEPS` stop by optimizer steps.
- `TRAIN_MAX_MICROSTEPS` is an optional hard stop for old-style microstep caps.
- `input_tokens_seen` counts all tokens fed to the model.
- `loss_tokens_seen` counts only labels that are not masked with `-100`.

Recommended stages:

1. `causal_lm_full` on curated Cosmopedia `text`-only public pretraining.
2. `assistant_masked_sft` on first-party XML action traces.
3. Optional later preference training after both objective gates pass.

The accepted runtime artifact is the SFT-stage export. A pretrain-only artifact
is not accepted for chat, even when fixed artifact checks pass.

## Artifacts

- Datasets: `data/train/datasets`
- Active full corpus: `data/public-corpus`
- Active first-party SFT corpus: `corpus/generated/kimi-full-v1` until a
  refreshed 60M validated corpus is generated.
- Tokenizer: `data/train/tokenizer`
- Checkpoints: `data/train/checkpoints`
- Native exports: `data/models/lkjai-scratch-40m`
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
