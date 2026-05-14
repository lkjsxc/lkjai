# Quickstart

Owner: `docs/getting-started/quickstart.md`.
State: canonical documentation.


## Goal

Bring up the direct OpenAI-compatible inference route and run the scratch
training path with Docker Compose.

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
  data/models/dense-diagnostic-scratch-40m \
  data/models/decoder-40m-3070 \
  data/train data/agent data/workspace
```

## Run OpenAI-Compatible Inference

For chat, `.env` must select an existing decoder export under
`data/models/${MODEL_NAME}`:

```dotenv
MODEL_NAME=decoder-40m-3070
```

Then start the API-only inference profile:

```bash
docker compose --profile inference up --build -d
```

OpenAI-compatible routes:

- `curl --fail http://127.0.0.1:8081/v1/models`
- `http://127.0.0.1:8081/v1/chat/completions`

Minimal chat probe:

```bash
curl -sS -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "decoder-40m-3070",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Decoder artifacts with the real local byte-level BPE `tokenizer.json` return
OpenAI-compatible `choices`. Dense and transformer artifacts start and report
readiness when loadable, but `/v1/chat/completions` returns HTTP `422` with no
`choices`. Missing artifacts make `GET /v1/models` return HTTP `503`.

The default `MODEL_NAME=decoder-40m-3070` expects a decoder export. Dense
artifacts can still start, while chat honestly reports unsupported decode.

## Run Web Runtime

```bash
docker compose --profile web up --build web
```

Start the sandbox API beside inference for chat:

```bash
docker compose --profile sandbox up --build -d
```

This starts three loopback services:

- `web`: static frontend on `http://127.0.0.1:8080`.
- `sandbox`: native agent API on `http://127.0.0.1:8082`.
- `inference`: OpenAI-compatible model API on `http://127.0.0.1:8081`.

Web app endpoint:

- `http://127.0.0.1:8080`

Route checks:

- `curl --fail http://127.0.0.1:8082/healthz`
- `curl --fail http://127.0.0.1:8082/api/config`
- `curl --fail http://127.0.0.1:8081/v1/models`

The sandbox calls the inference service for `/api/chat`. Dense and transformer
artifacts return HTTP `422` with no `choices` for `/v1/chat/completions`.
Decoder artifacts can return chat choices only when the artifact includes the
real local byte-level BPE `tokenizer.json`.

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

This starts the decoder 40M RTX 3070 training profile with a two-hour
wall-clock target. It fails fast if the packed cache is missing, incompatible,
or still the seq16/vocab256 smoke fixture.

For a 40M Docker start check after the cache exists:

```bash
docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/train-start-check \
  -e TRAIN_MAX_OPTIMIZER_STEPS=1 \
  -e TRAIN_MAX_STEPS=1 \
  train --train
```

Expected training artifacts:

- `data/train/tokenizer/`: local byte-level BPE tokenizer.
- `data/train/checkpoints/final/`: native decoder checkpoint.
- `data/train/runs/train-report.json`: native train report.
- `data/models/decoder-40m-3070/`: serving artifact export.

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
