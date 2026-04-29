# XML Action Repair

## Goal

Create a short, focused SFT pass that teaches the model to emit complete XML
actions without relying on runtime fallback wrapping.

## Problem

The `270000`-step overnight SFT checkpoint is structurally protected by runtime
fallback, but raw model behavior is still poor:

- Generation sanity passes only because invalid output becomes `agent.finish`.
- Behavioral eval remains `0/200`.
- Raw next-token probabilities after `<assistant_action>` do not reliably start
  with `<action>`.

## Repair Strategy

Use a deterministic local repair corpus under `data/xml-action-repair-v1/`.
Rows are ordinary supervised conversations whose assistant targets are complete
XML actions:

- Direct everyday replies use `agent.finish`.
- Workspace inspection requests use filesystem tools.
- Memory requests use memory tools.
- Destructive or write-like requests use `agent.request_confirmation`.
- Tool observations are followed by final `agent.finish` answers.

The corpus is not an accepted substitute for behavioral eval. It is a repair
curriculum to make the model itself produce XML before the fallback path.

## Run

```bash
docker compose --profile train run --rm --entrypoint python train \
  -m lkjai_train.repair_corpus /app/data/xml-action-repair-v1

docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/train-xml-repair-v1 \
  -e TRAIN_COMMITTED_CORPUS_DIR=/app/data/xml-action-repair-v1 \
  -e TRAIN_INIT_CHECKPOINT=/app/data/train-full-500m-from-scratch-v2/checkpoints/best \
  -e TRAIN_RESUME=never \
  -e TRAIN_MAX_OPTIMIZER_STEPS=120000 \
  train train-sft --preset agent
```

## Acceptance

After repair training:

- Export the best checkpoint.
- Run generation sanity.
- Run behavioral eval.
- Manually test lkjai web with `Hello`, `Thanks`, `List files`, and memory
  preference prompts.
- Inspect raw generated text; do not count fallback output as success.
