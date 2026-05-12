# Quickstart

## Goal

Bring up the local agent runtime and run the scratch training path with Docker
Compose.

## Prerequisites

- Docker Engine + Compose plugin.
- NVIDIA driver + NVIDIA container runtime for training runs.
- RTX 3070 8GB target machine.
- Free disk for model, tokenizer, checkpoint, memory, and transcript artifacts
  under `data/`.

## Prepare Workspace

```bash
cp .env.example .env
mkdir -p \
  data/models/lkjai-scratch-40m \
  data/models/dense-40m-3070 \
  data/train data/agent data/workspace
```

## Run Web Runtime

```bash
docker compose --profile web up --build web
```

This starts one merged native server:

- `web`: native C++ agent API runtime and OpenAI-compatible model routes.

Web app endpoint:

- `http://127.0.0.1:8080`

Merged model route checks:

- `curl --fail http://127.0.0.1:8080/v1/models`
- `http://127.0.0.1:8080/v1/chat/completions`

The merged implementation loads exported scratch artifacts and exposes
readiness plus logits-oriented native checks. Dense and transformer artifacts
return HTTP `422` with no `choices` for `/v1/chat/completions`. Decoder
artifacts can return chat choices only when the artifact includes the real local
byte-level BPE `tokenizer.json`.

## Run Inference Alone

```bash
docker compose --profile inference up --build inference
```

Use this only when probing model routes without the web UI.

## Run Scratch Training

Build the required tokenizer and packed cache first:

```bash
docker compose --profile corpus run --build --rm corpus build-tokenizer
docker compose --profile corpus run --rm corpus validate-public-pretrain
docker compose --profile corpus run --rm corpus build-public-pretrain-cache
```

```bash
docker compose --profile train up --build train
```

This starts the dense 40M RTX 3070 training profile with a two-hour
wall-clock target. It fails fast if the packed cache is missing, incompatible,
or still the seq16/vocab256 smoke fixture.

For a 40M Docker start check after the cache exists:

```bash
docker compose --profile train run --rm \
  -e TRAIN_DATA_DIR=/app/data/train-start-check \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_MAX_STEPS=1 \
  train --train
```

Expected training artifacts:

- `data/train/tokenizer/`: local byte-level BPE tokenizer.
- `data/train/checkpoints/final/`: native dense checkpoint.
- `data/train/runs/train-report.json`: native train report.
- `data/train/exports/dense-40m-3070/`: serving artifact export.

Behavioral chat evaluation is meaningful only for decoder artifacts. Dense
exports remain training and diagnostics artifacts.

## Inspect Runtime Outputs

- `data/agent/runs/`: chat run transcripts.
- `data/agent/runs/`: durable JSONL transcript and event history.
- `data/workspace/`: only filesystem root used by agent file and shell tools.
- `GET /api/model`: active model client status and reachability.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common failures.

## Required Verification Before Commit

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```
