# Training Pipeline

## Goal

Run real adapter training and produce measurable artifacts for local serving and
evaluation.

## Commands

- `docker compose --profile train up --build train`.
- `python -m lkjai_train.cli prepare-fixtures`.
- `python -m lkjai_train.cli prepare-corpus`.
- `python -m lkjai_train.cli validate-dataset`.
- `python -m lkjai_train.cli train-adapter`.
- `python -m lkjai_train.cli fixed-eval`.
- `python -m lkjai_train.cli export-manifest`.
- `python -m lkjai_train.cli smoke`.

## Pipeline Order

1. Prepare built-in starter dataset or generate synthetic corpus.
2. Validate message/tool trajectory schema and split metadata.
3. Run QLoRA adapter training with Transformers + PEFT + bitsandbytes.
4. Persist adapter checkpoints and train metadata.
5. Run fixed evals and write report under `data/train/runs/`.
6. Export manifest metadata for serving handoff.
7. Keep deterministic `smoke` command for CI-only fast checks.

## Dataset Preparation

- `prepare-fixtures`: writes 2 deterministic rows for smoke tests.
- `prepare-corpus`: generates >= 100 synthetic agent trajectories.
- Both write JSONL with `messages` and `tags`.
- Chat template formatting happens during tokenization, not at dataset write
  time.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_PRESET=agent` is the Compose default.
- `TRAIN_PRESET=quick` runs real training with reduced steps for fast checks.
- `TRAIN_PRESET=custom` reads explicit `TRAIN_*` knobs.
- `TRAIN_DATA_DIR` defaults to `/app/data/train`.

## Presets

| Preset | Base Model | Steps | Corpus Size | Purpose |
|--------|-----------|-------|-------------|---------|
| quick  | Qwen/Qwen3-0.6B | 50 | 20 | CI smoke |
| agent  | Qwen/Qwen3-0.6B | 500 | 200 | Real tuning |
| custom | `TRAIN_BASE_MODEL` | `TRAIN_MAX_STEPS` | `TRAIN_CORPUS_SIZE` | Full control |

## Artifacts

- Datasets live under `data/train/datasets`.
- Adapters live under `data/train/adapters`.
- Exports live under `data/train/exports`.
- Eval reports live under `data/train/runs`.

## Verification

```bash
docker compose --profile train up --build train
ls -lh data/train/adapters/final/
cat data/train/runs/fixed-eval.json | jq .pass_rate
```

Expected: adapter directory contains `.safetensors` or `.bin` files;
`pass_rate` >= threshold.
