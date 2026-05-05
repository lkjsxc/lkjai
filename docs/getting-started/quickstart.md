# Quickstart

## Goal

Bring up the local agent runtime and run the scratch training path with Docker
Compose.

## Prerequisites

- Docker Engine + Compose v2.
- NVIDIA driver + NVIDIA container runtime for training runs.
- RTX 3070 8GB target machine.
- Free disk for model, tokenizer, checkpoint, memory, and transcript artifacts
  under `data/`.

## Prepare Workspace

```bash
cp .env.example .env
mkdir -p data/models/lkjai-scratch-40m data/train data/agent data/workspace
```

## Run Web Runtime

```bash
docker compose --profile web up --build web
```

This starts both containers:

- `inference`: scratch OpenAI-compatible model server.
- `web`: native C++ agent API runtime.

Web app endpoint:

- `http://127.0.0.1:8080`

Inference API endpoint:

- `http://127.0.0.1:8081/v1/chat/completions`
- `curl --fail http://127.0.0.1:8081/v1/models`

The inference implementation loads exported scratch artifacts and exposes
readiness plus logits-oriented native checks. Dense and transformer artifacts
return HTTP `422` with no `choices` for `/v1/chat/completions`. Decoder
artifacts can return chat choices only when the artifact includes the real local
byte-level BPE `tokenizer.json`.

## Run Inference Alone

```bash
docker compose --profile inference up --build inference
```

Use this only when probing the model server without the web UI.

## Run Scratch Training

```bash
docker compose --profile train up --build train
```

For a quick smoke check:

```bash
TRAIN_PRESET=quick docker compose --profile train up --build train
```

For a 40M Docker start check without the full long run:

```bash
docker compose --profile train run --rm \
  -e TRAIN_DATA_DIR=/app/data/train-start-check \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_MAX_STEPS=1 \
  -e TRAIN_RESUME=never \
  train --train
```

Expected training artifacts:

- `data/train/tokenizer/`: local byte-level BPE tokenizer.
- `data/train/checkpoints/final/`: native dense checkpoint.
- `data/train/runs/train-report.json`: native train report.
- `data/train/exports/lkjai-scratch-40m/`: serving artifact export.

Behavioral chat evaluation is meaningful only for decoder artifacts. Dense
exports remain training and diagnostics artifacts.

## Inspect Runtime Outputs

- `data/agent/runs/`: chat run transcripts.
- `data/agent/memory.sqlite3`: durable memory database.
- `data/workspace/`: only filesystem root used by agent file and shell tools.
- `GET /api/model`: active model client status and reachability.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common failures.

## Required Verification Before Commit

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```
