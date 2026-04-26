# Monitoring and Health

## Goal

Observe runtime health without adding heavy telemetry dependencies.

## Contract

- The web runtime must verify inference is reachable before claiming a model is
  loaded.
- Compose inference health uses `/healthz` so the web UI can start even before
  a model export exists.
- Model readiness failures must be exposed through `GET /api/model` and the
  web UI.
- The runtime must never silently fall back to canned or fake responses when
  inference is unreachable.

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
  "model": "lkjai-scratch-40m",
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

Expected: `reachable` matches the actual inference server state.
