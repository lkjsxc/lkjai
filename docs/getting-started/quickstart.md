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
- `web`: Rust agent and browser UI.

Web app endpoint:

- `http://127.0.0.1:8080`

Inference API endpoint:

- `http://127.0.0.1:8081/v1/chat/completions`
- `curl --fail http://127.0.0.1:8081/v1/models`

The inference implementation loads exported scratch artifacts and generates
XML actions from the trained native checkpoint.

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
  -e TRAIN_COMPILE=off \
  train train-scratch --preset agent
```

Expected training artifacts:

- `data/train/tokenizer/`: local byte-level BPE tokenizer.
- `data/train/checkpoints/final/`: scratch model weights and config.
- `data/train/runs/fixed-eval.json`: evaluation report.
- `data/train/runs/behavioral-eval.json`: generated response competency report.
- `data/train/exports/manifest.json`: serving metadata.

## Inspect Runtime Outputs

- `data/agent/runs/`: chat run transcripts.
- `data/agent/memory.sqlite3`: durable memory database.
- `data/workspace/`: only filesystem root used by agent file and shell tools.
- `GET /api/model`: active model client status and reachability.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common failures.

## Required Verification Before Commit

```bash
docker compose --profile verify up --build --abort-on-container-exit verify
```
