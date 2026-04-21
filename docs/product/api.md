# API Contract

## Routes

- `GET /`: chat UI.
- `GET /healthz`: returns `200` with body `ok`.
- `POST /api/chat`: runs one chat turn and optional tool actions.
- `GET /api/runs/{id}`: returns one run transcript.

## `POST /api/chat` Request

```json
{
  "message": "string",
  "run_id": "optional-string"
}
```

## `POST /api/chat` Response

```json
{
  "run_id": "string",
  "assistant": "string",
  "events": []
}
```

## Event Shape

- `kind`: `user`, `assistant`, `tool_call`, `tool_result`, or `error`.
- `content`: human-readable content.
- `tool`: optional tool name.
- `timestamp`: RFC 3339 timestamp.
