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

## Train Smoke Model

```bash
docker compose --profile train run --rm train smoke
```

## Run Web App

```bash
docker compose --profile web up --build web
```

The default web bind is `127.0.0.1:${APP_PORT}`.
