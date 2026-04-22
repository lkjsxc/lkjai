# Quickstart

## Prerequisites

- Docker with Compose v2.
- NVIDIA driver and NVIDIA container runtime for the model and train profiles.
- Enough disk under `data/` for GGUF models, adapters, transcripts, and memory.

## Prepare

```bash
cp .env.example .env
mkdir -p data
```

## Verify

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Run Web App With Trained Policy

```bash
docker compose --profile web up --build
```

The web profile starts the Rust orchestrator and loads the trained policy from
`data/train/policy/model.json`.

## Run Offline Verification

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

Verification uses deterministic fixtures and does not download large models.

## Train Agent Adapter

```bash
docker compose --profile train up --build train
```

The default train preset is a quick schema/eval path. Use
`TRAIN_PRESET=agent` for RTX 3070 QLoRA tuning.

## Inspect Outputs

- `data/agent/runs/`: chat and tool transcripts.
- `data/agent/memory.sqlite3`: durable memory.
- `data/train/runs/fixed-eval.json`: agent eval report.
- `data/train/adapters/`: QLoRA adapters.
