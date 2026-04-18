# Private Console Contract

## Goal

Provide a single-operator interface for chatting with the agent and managing records.

## Routes

- `GET /`: private console HTML.
- `GET /healthz`: runtime health check.
- `POST /api/chat`: send chat prompts to the agent.
- `POST /api/records/upsert`: create or update records.
- `POST /api/records/delete`: delete records by id.
- `GET /api/records/list`: list records with optional query.

## Access Rules

- All `/api/*` endpoints require admin token authentication.
- Token is supplied via `x-admin-token` header.
- Authentication failures return `401` JSON responses.

## UX Rules

- Chat input and responses stay in one scrolling timeline.
- Record operations return structured status and metadata.
- Errors surface explicit machine-readable codes.

