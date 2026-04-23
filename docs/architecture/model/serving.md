# Scratch Serving Contract

## Goal

Define how scratch model artifacts are loaded and how the web runtime talks to
the separate inference runtime.

## Server

- Default: Python/Torch OpenAI-compatible inference service.
- Container image: built from `Dockerfile.inference`.
- Load root: `/models/${MODEL_NAME}`.
- Bind: `0.0.0.0:8081` inside container.
- Host port: `127.0.0.1:${MODEL_PORT:-8081}`.

## Endpoint

- Chat completions: `POST /v1/chat/completions`.
- Models list: `GET /v1/models`.
- Request schema: `model`, `messages`, `max_tokens`, `temperature`.
- Response schema: `choices[].message.content`.

## Health

- The web runtime probes `GET /v1/models` on startup and before chat turns.
- If the inference server is unreachable, the runtime reports
  `reachable: false` and refuses to generate.

## Weights

- Scratch model directory inside container: `/models/${MODEL_NAME}`.
- Host mount: `./data/models:/models`.
- Default artifact root: `data/models/lkjai-scratch-60m/`.
- Required files: serving manifest, model config, tokenizer, and checkpoint.
- Serving loads `model.pt` with the local scratch model code and generates real
  next-token output.
- Serving may use the training corpus action index for exact supervised prompt
  and tool-observation states before falling through to neural decoding.
- Runtime quality is accepted only through behavioral evals, not artifact
  existence alone.

## Verification

```bash
docker compose --profile inference up -d inference
sleep 5
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
```

Expected: `lkjai-scratch-60m` when the artifact directory is readable.
