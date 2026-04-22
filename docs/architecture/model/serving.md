# Model Serving Contract

## Goal

Define how model weights are loaded and how the runtime talks to the model
server.

## Server

- Default: llama.cpp OpenAI-compatible CUDA server.
- Container image: `ghcr.io/ggml-org/llama.cpp:server-cuda`.
- Load command: `-m /models/${MODEL_GGUF}`.
- Bind: `0.0.0.0:8080` inside container.
- Host port: `127.0.0.1:${MODEL_PORT:-8081}`.

## Endpoint

- Chat completions: `POST /v1/chat/completions`.
- Models list: `GET /v1/models`.
- Request schema: `model`, `messages`, `max_tokens`, `temperature`.
- Response schema: `choices[].message.content`.

## Health

- The web runtime probes `GET /v1/models` on startup and before chat turns.
- If the model server is unreachable, the runtime reports `reachable: false` and
  refuses to generate.

## Weights

- GGUF file path inside container: `/models/${MODEL_GGUF}`.
- Host mount: `./data/models:/models`.
- Default artifact: `data/models/qwen3-1.7b-q4.gguf`.

## Verification

```bash
docker compose --profile model up -d model
sleep 5
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
```

Expected: a model identifier string.
