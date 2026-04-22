# Monitoring and Health

## Goal

Observe runtime health without adding heavy telemetry dependencies.

## Contract

- The web runtime must verify the model server is reachable before claiming it is
  loaded.
- The model server health probe must use the OpenAI-compatible models endpoint or
  a lightweight server-specific health endpoint.
- Health failures must be exposed to operators through `GET /api/model` and the
  web UI.
- The runtime must never silently fall back to canned or fake responses when the
  model server is unreachable.

## Health Probe

```
GET ${MODEL_API_URL%/v1/chat/completions}/models
```

- Success: HTTP 200 with a JSON body containing at least one model id.
- Failure: any non-2xx status, timeout, or connection error.
- Timeout: 5 seconds.

## Model Status Response

```json
{
  "model": "qwen3-1.7b-q4",
  "api_url": "http://127.0.0.1:8081/v1/chat/completions",
  "loaded": true,
  "reachable": true,
  "message": "model server responding"
}
```

- `loaded`: the client is configured with a model name and URL.
- `reachable`: the last health probe succeeded.
- `message`: human-readable state.

## UI Behavior

- When `reachable` is `false`, the UI displays "model unavailable" and disables
  the send button.
- When `reachable` becomes `true`, the UI resumes normal operation.

## Verification

```bash
curl -sf http://127.0.0.1:8080/api/model | jq .
```

Expected: `reachable` matches the actual model server state.
