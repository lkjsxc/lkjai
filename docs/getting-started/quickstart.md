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
mkdir -p data/models/lkjai-scratch-60m data/train data/agent data/workspace
```

## Run Inference Service

```bash
docker compose --profile inference up --build inference
```

Inference API endpoint:

- `http://127.0.0.1:8081/v1/chat/completions`
- `curl --fail http://127.0.0.1:8081/v1/models`

The inference implementation loads exported scratch artifacts and generates
JSON actions from the trained PyTorch checkpoint.

## Run Web Runtime

```bash
docker compose --profile web up --build web
```

Web app endpoint:

- `http://127.0.0.1:8080`

## Run Scratch Training

```bash
docker compose --profile train up --build train
```

For a quick smoke check:

```bash
TRAIN_PRESET=quick docker compose --profile train up --build train
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
