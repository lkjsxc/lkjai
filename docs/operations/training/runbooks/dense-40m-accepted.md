# Dense 40M Accepted Runbook

Owner: `docs/operations/training/runbooks/dense-40m-accepted.md`.
State: canonical dense 40M browser-demo runbook.

## Goal

Promote the dense 40M native path as the accepted browser-demo target. This
runbook does not claim decoder chat, full decoder backward, or CUDA KV-cache
decode.

## Config Pair

- Training config:
  `configs/training/dense_40m_accepted_3070.json`.
- Native config:
  `configs/native/native_dense_40m_bf16_3070.json`.
- Model name: `dense-diagnostic-40m-3070`.
- Local route claim: none. Dense diagnostics are native helper checks; sandbox
  `/api/*` is reserved for agent runtime routes.

## Prerequisites

- `docker compose --progress quiet --profile verify run --build --rm verify`
  passes before the long run.
- The packed cache at
  `data/train/datasets/packed/train-causal_lm_full-seq1024` validates against
  the same vocab, sequence length, tokenizer, and source manifest.
- CUDA BF16 and cuBLASLt capability checks pass on the target machine.

## Run

```bash
docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/dense-40m-accepted \
  -e TRAIN_CONFIG=/workspace/configs/training/dense_40m_accepted_3070.json \
  -e TRAIN_NATIVE_CONFIG=/workspace/configs/native/native_dense_40m_bf16_3070.json \
  train
```

The config uses warmup plus cosine decay, `target_seconds=7200`, and
`save_latest_every_optimizer_steps=512`.

## Required Outputs

- `checkpoints/latest/`
- `checkpoints/best/`
- `checkpoints/final/`
- `exports/dense-diagnostic-40m-3070/`
- `../models/dense-diagnostic-40m-3070/`
- `runs/train-report.json`

The report must include `stop_reason`, `deadline_hit`, `lr_schedule`,
`min_learning_rate_fraction`, `best_checkpoint_path`,
`best_checkpoint_checksum`, `optimizer_steps`, `loss`, `parameter_count`,
`checkpoint_checksum`, `export_checksum`, and `logits_checksum`.

## Evidence

Commit only tracked evidence summaries after validating the generated artifacts.
Do not commit model weights, optimizer state, raw packed-cache data, or
generated CUDA profiling blobs.
