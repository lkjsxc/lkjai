# API Contract

## Routes

- `GET /`: chat UI.
- `GET /healthz`: returns `200` with body `ok`.
- `POST /api/chat`: runs one bounded agent turn.
- `GET /api/runs/{id}`: returns one run transcript.
- `GET /api/model`: returns model client status including reachability.

## `POST /api/chat` Request

```json
{
  "message": "string",
  "run_id": "optional-string",
  "max_steps": 6
}
```

## `POST /api/chat` Response

```json
{
  "run_id": "string",
  "assistant": "string",
  "events": [],
  "stop_reason": "finish"
}
```

## `GET /api/model` Response

```json
{
  "model": "lkjai-scratch-60m",
  "api_url": "http://127.0.0.1:8081/v1/chat/completions",
  "loaded": true,
  "reachable": true,
  "message": "model server responding"
}
```

- `loaded`: client is configured.
- `reachable`: last health probe succeeded.

## Event Shape

- `kind`: `user`, `assistant`, `reasoning`, `plan`, `tool_call`,
  `tool_result`, `observation`, `memory_write`, `finish`,
  `confirmation_request`, or `error`.
- `content`: human-readable content.
- `tool`: optional tool name.
- `timestamp`: RFC 3339 timestamp.
- `step`: optional agent loop step.

`reasoning` events come from the model's `<reasoning>` child tag. They are
visible brief rationales and must not contain hidden chain-of-thought detail.

## Error Contract

- Invalid model responses must produce `error` events in `events`.
- If the model server is unreachable, `stop_reason` is `model_error`.
- If no final assistant action is produced, `stop_reason` must indicate failure.
- `GET /api/model` reflects runtime model client configuration and reachability,
  not benchmarked quality.

## Verification

```bash
curl -sf http://127.0.0.1:8080/healthz
curl -sf http://127.0.0.1:8080/api/model | jq .
curl -sf -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}' | jq '.stop_reason'
```
