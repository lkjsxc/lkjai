# kjxlkj Integration

Owner: `docs/product/kjxlkj-integration.md`.
State: canonical documentation.


## Goal

Make `lkjai` ready to act as the server-side assistant for `kjxlkj` through
typed resource APIs.

## Canonical Tool Surface

- `resource.search`
- `resource.fetch`
- `resource.history`
- `resource.preview_markdown`
- `resource.create_note`
- `resource.create_media`
- `resource.update_resource`

## Rules

- `lkjai` should target `kjxlkj` resource APIs, not filesystem-shaped note
  folders.
- `lkjai` uses `Authorization: Bearer <token>` with `KJXLKJ_USER` selecting
  the personal space.
- Browser session cookies are not part of the integration contract.
- Read, search, history, and preview may run directly.
- Create and update operations must first produce
  `{"kind":"request_confirmation", ...}` and must not execute until the next
  user turn explicitly confirms the pending operation.
- The mainline integration is API and contract work only. No end-user chat UI
  is required in this pass.
- `GET /api/config` must expose the configured `KJXLKJ_API_URL`, `KJXLKJ_USER`,
  bearer-token presence, and `/api/users/{user}/resources` base URL. It must
  report degraded status when the bearer token is absent.

## Route Mapping

- `resource.search` -> `GET /api/users/{user}/resources/search`
- `resource.fetch` -> `GET /api/users/{user}/resources/{ref}`
- `resource.history` -> `GET /api/users/{user}/resources/{ref}/history`
- `resource.preview_markdown` -> `POST /api/users/{user}/resources/preview-markdown`
- `resource.create_note` -> `POST /api/users/{user}/resources/notes`
- `resource.create_media` -> `POST /api/users/{user}/resources/media`
- `resource.update_resource` -> `PUT /api/users/{user}/resources/{ref}`

## Request Examples

Read-only search:

```http
GET /api/users/default/resources/search?q=training%20report&limit=5
Authorization: Bearer ${KJXLKJ_BEARER_TOKEN}
```

Fetch by resource reference:

```http
GET /api/users/default/resources/note_123
Authorization: Bearer ${KJXLKJ_BEARER_TOKEN}
```

Preview a markdown update before confirmation:

```http
POST /api/users/default/resources/preview-markdown
Authorization: Bearer ${KJXLKJ_BEARER_TOKEN}
Content-Type: application/json

{"body":"# Draft\n\nProposed content."}
```

## Degraded Behavior

- If `KJXLKJ_BEARER_TOKEN` is empty, `GET /api/config` reports degraded status
  and mutable resource tools remain disabled.
- Read-only resource tools may return configuration errors, but the runtime
  must not invent resource results.
- Mutation tools remain disabled unless `AGENT_TOOL_PROFILE=mutable`, bearer
  token presence is confirmed, and a previous turn produced a confirmation
  request.
