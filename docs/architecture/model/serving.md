# Scratch Serving Contract

## Goal

Serve exported scratch checkpoints through one raw-generation path that matches
evaluation.

## Server

- Default backend: Python/Torch OpenAI-compatible inference service.
- Container image: `ops/docker/Dockerfile.inference`.
- Load root: `/models/${MODEL_NAME}`.
- Bind: `0.0.0.0:8081` in-container.
- Host port: `127.0.0.1:${MODEL_PORT:-8081}`.

## Endpoint

- `POST /v1/chat/completions`
- `GET /v1/models`
- Request fields: `model`, `messages`, `max_tokens`, `temperature`.
- Response field consumed by the Rust runtime: `choices[0].message.content`.

## Runtime Rules

- Required files: `manifest.json`, `config.json`, `tokenizer.json`, `model.pt`.
- Serving loads the exported checkpoint and generates tokens directly.
- Decode must reuse KV cache across generated tokens.
- No supervised action index, prompt lookup table, or policy-file fallback is
  allowed in the accepted runtime path.
- Adapter seams are allowed for future backends, but the 3070-first backend is
  the Python/Torch path.

## Health

- The web runtime probes `GET /v1/models`.
- If the model is unreachable, chat stops with `model_error`.

## Verification

```bash
docker compose --profile inference up -d --build inference
sleep 5
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
```

Expected: `lkjai-scratch-40m` when exported artifacts are readable.
