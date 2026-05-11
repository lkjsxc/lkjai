# Monitoring and Health

## Goal

Observe runtime health without adding heavy telemetry dependencies.

## Contract

- The merged server must verify the loaded artifact before claiming a model is
  ready.
- Compose health uses `/healthz` so clients can see missing model exports.
- Model readiness failures must be exposed through `GET /api/model` and the
  web UI.
- The runtime must never silently fall back to canned or fake responses when the
  model is unavailable.

## Health Probe

```
GET /v1/models
```

- Success: HTTP 200 with a JSON body containing at least one model id.
- Failure: any non-2xx status, timeout, or connection error.
- Timeout: 5 seconds.

## Model Status Response

```json
{
  "model": "lkjai-scratch-40m",
  "api_url": "local-native-engine",
  "loaded": true,
  "reachable": true,
  "message": "model loaded"
}
```

- `loaded`: the artifact loaded successfully.
- `reachable`: the local model engine is usable.
- `message`: human-readable state.
- `probe_status`: `200` when the artifact is loaded, otherwise `503`.

## UI Behavior

- When `reachable` is `false`, the UI displays "model unavailable" and disables
  the send button.
- When `reachable` becomes `true`, the UI resumes normal operation.

## Verification

```bash
curl -sf http://127.0.0.1:8080/api/model | jq .
```

Expected: `reachable` matches the loaded artifact state.
