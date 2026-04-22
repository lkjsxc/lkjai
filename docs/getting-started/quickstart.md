# Quickstart

## Goal

Bring up a local, real-LLM runtime and execute a real training run with Docker
Compose.

## Prerequisites

- Docker Engine + Compose v2.
- NVIDIA driver + NVIDIA container runtime.
- RTX 3070 8GB target machine.
- Free disk for model weights and training artifacts under `data/`.

## Prepare Workspace

```bash
cp .env.example .env
mkdir -p data/models data/train data/agent
```

## Put a GGUF Model in Place

Place your serving model at:

- `data/models/${MODEL_GGUF}`

Default in `.env.example`:

- `MODEL_GGUF=qwen3-1.7b-q4.gguf`

## Run Model Service

```bash
docker compose --profile model up -d model
```

Model API endpoint (default):

- `http://127.0.0.1:8081/v1/chat/completions`

## Run Web Runtime

```bash
docker compose --profile web up --build web
```

Web app endpoint (default):

- `http://127.0.0.1:8080`

## Run Real Training

```bash
TRAIN_PRESET=agent docker compose --profile train up --build train
```

Expected training artifacts:

- `data/train/adapters/`: adapter checkpoints and metadata.
- `data/train/runs/fixed-eval.json`: evaluation report.
- `data/train/exports/manifest.json`: export metadata.

## Inspect Runtime Outputs

- `data/agent/runs/`: chat run transcripts.
- `data/agent/memory.sqlite3`: durable memory database.
- `GET /api/model`: active model client status.

## Required Verification Before Commit

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```
