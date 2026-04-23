# Model Client Runtime

## Goal

Call a separate OpenAI-compatible inference endpoint with live health
verification.

## Contract

- The web runtime must call the separate inference endpoint; no fake or
  policy-file fallback is allowed in production.
- The runtime must verify the inference server is reachable before serving chat
  requests.
- Health probe uses `GET /v1/models` with a 5-second timeout.
- `GET /api/model` returns `reachable` based on the last probe result.

## Request/Response

- Request schema: `model`, `messages`, `max_tokens`, `temperature`.
- Response schema: first `choices[].message.content` is consumed.
- Each model step must return strict JSON action text for the agent parser.
- Initial scratch inference may return deterministic JSON while real decode is
  incomplete, but it must still be served by the inference container.

## Failure Semantics

- HTTP request failures surface as explicit transcript error events with stop
  reason `model_error`.
- Non-success status from inference includes status code and body text.
- Parse failures from model response surface as explicit transcript errors.
- No canned fallback answer is allowed inside the web runtime.
- If inference is unreachable, the agent loop stops with `model_error`.

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=lkjai-scratch-40m`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`

## Verification

```bash
curl -sf http://127.0.0.1:8081/v1/models | jq '.data[0].id'
curl -sf http://127.0.0.1:8080/api/model | jq '.reachable'
```

Expected: model id is a non-empty string; `reachable` is `true` when inference
is up.
