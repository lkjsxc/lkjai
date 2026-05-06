# Scratch Serving Contract

Owner: `docs/architecture/model/serving.md`.
State: canonical model serving capability contract.

## Goal

Load exported scratch artifacts through one native service. Current dense
artifacts prove load and logits, while raw generation belongs to the native
`decoder` milestone.

## Server

- Default backend: native OpenAI-compatible inference service.
- Container image: `ops/docker/Dockerfile.native`.
- Load root: `/models/${MODEL_NAME}`.
- Bind: `0.0.0.0:8081` in-container.
- Host port: `127.0.0.1:${MODEL_PORT:-8081}`.

## Endpoint

- `POST /v1/chat/completions`
- `GET /v1/models`
- Request fields: `model`, `messages`, `max_tokens`, `temperature`.
- Successful decoder responses expose `choices[0].message.content`.
- Current dense and transformer artifacts return HTTP `422` unsupported decode
  with no `choices` field.
- Decoder artifacts with the real local tokenizer may return `choices`; dense
  and transformer artifacts must not.

## Runtime Rules

- Required files: `manifest.json`, `config.json`, `tokenizer.json`,
  `weights.index.json`, and `weights.lkjw`.
- Serving currently loads native exports and exposes readiness. Dense and
  transformer artifacts are diagnostics; decoder artifacts may return partial
  host-reference chat choices until accepted KV-cache decode lands.
- `lkjai-native-logits-check` is the accepted current inference proof.
- Target decode must reuse preallocated KV cache storage across generated
  tokens.
- Target stop detection uses tokenizer ids for canonical XML tags and decodes
  text once after generation.
- No supervised action index, prompt lookup table, or policy-file fallback is
  allowed in the accepted runtime path.
- If decode cannot produce one complete `</action>`, the server returns a
  non-success response instead of wrapping the failure in a valid action.
- Adapter seams are allowed for future backends, but the 3070-first backend is
  the native path.

## Health

- Compose probes `GET /healthz` for process health.
- The web runtime probes `GET /v1/models` for model readiness.
- If the model is unreachable, chat stops with `model_error`.

## Verification

```bash
docker compose --profile inference up -d --build inference
sleep 5
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
```

Expected: `lkjai-scratch-40m` when exported artifacts are readable.
