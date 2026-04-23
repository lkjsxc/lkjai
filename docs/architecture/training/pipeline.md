# Scratch Training Pipeline

## Goal

Train tokenizer, corpus, scratch language-model checkpoints, and agent-style
supervision artifacts from local data.

## Commands

- `docker compose --profile train up --build train`.
- `python -m lkjai_train.cli prepare-fixtures`.
- `python -m lkjai_train.cli prepare-corpus`.
- `python -m lkjai_train.cli train-tokenizer`.
- `python -m lkjai_train.cli validate-dataset`.
- `python -m lkjai_train.cli train-scratch`.
- `python -m lkjai_train.cli fixed-eval`.
- `python -m lkjai_train.cli export-manifest`.
- `python -m lkjai_train.cli smoke`.

## Pipeline Order

1. Prepare built-in starter dataset or generate synthetic corpus.
2. Train a local byte-level BPE tokenizer from corpus text.
3. Validate message/tool trajectory schema and split metadata.
4. Initialize a small dense decoder from random weights.
5. Train next-token prediction on structured chat/action trajectories.
6. Persist scratch checkpoints and train metadata.
7. Run fixed evals and write report under `data/train/runs/`.
8. Export manifest metadata for serving handoff.
9. Keep deterministic `smoke` command for fast local checks.

## Dataset Preparation

- `prepare-fixtures`: writes 2 deterministic rows for smoke tests.
- `prepare-corpus`: generates synthetic agent trajectories.
- Both write JSONL with `messages` and `tags`.
- Formatting happens through the project scratch chat serializer, not through an
  upstream pretrained tokenizer template.

## Compose Entrypoint

- The train image defaults to `python -m lkjai_train.cli train`.
- `TRAIN_PRESET=agent` is the Compose default.
- `TRAIN_PRESET=quick` runs tiny scratch training with reduced steps.
- `TRAIN_PRESET=custom` reads explicit `TRAIN_*` knobs.
- `TRAIN_DATA_DIR` defaults to `/app/data/train`.

## Presets

| Preset | Model | Steps | Corpus Size | Purpose |
|--------|-------|-------|-------------|---------|
| quick  | tiny scratch | 5 | 20 | local smoke |
| agent  | scratch-40m | 500 | 200 | main research path |
| custom | env config | `TRAIN_MAX_STEPS` | `TRAIN_CORPUS_SIZE` | full control |

## Artifacts

- Datasets live under `data/train/datasets`.
- Tokenizers live under `data/train/tokenizer`.
- Checkpoints live under `data/train/checkpoints`.
- Serving exports live under `data/train/exports`.
- Eval reports live under `data/train/runs`.

## Verification

```bash
docker compose --profile train up --build train
ls -lh data/train/checkpoints/final/
cat data/train/runs/fixed-eval.json | jq .pass_rate
```

Expected: checkpoint directory contains model weights, config, tokenizer
manifest, and `pass_rate` >= threshold.
