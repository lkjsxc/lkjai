# API Contract

## Routes

- `GET /`: chat UI.
- `GET /healthz`: returns `200` with body `ok`.
- `POST /api/chat`: runs one bounded agent turn.
- `GET /api/runs/{id}`: returns one run transcript.
- `GET /api/model`: returns model client status.

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
  "stop_reason": "final"
}
```

## Event Shape

- `kind`: `user`, `assistant`, `plan`, `tool_call`, `tool_result`,
  `observation`, `memory_write`, or `error`.
- `content`: human-readable content.
- `tool`: optional tool name.
- `timestamp`: RFC 3339 timestamp.
- `step`: optional agent loop step.
