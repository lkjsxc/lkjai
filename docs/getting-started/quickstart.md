# Quickstart

## Prerequisites

- Docker with Compose v2.
- NVIDIA driver and NVIDIA container runtime for CUDA profiles.
- Enough disk under `data/` for corpora, tokenized shards, checkpoints, and
  model exports.

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

## Train Default Long-Run Model

```bash
docker compose --profile train up --build
```

The Compose default is a duration-aware long run targeting ~6 hours.

## Train Quick Debug Model

```bash
TRAIN_PRESET=quick TRAIN_ENFORCE_COMPETENCY=0 docker compose --profile train up --build
```

## Run Web App

```bash
docker compose --profile web up --build web
```

The default web bind is `127.0.0.1:${APP_PORT}`.

## Inspect Training Outputs

- `data/train/runs/last-train.json`: latest training summary
- `data/train/runs/fixed-eval.json`: competency report (`pass_rate >= 0.80` required)
- `data/train/models/lkj-150m/`: serving export
