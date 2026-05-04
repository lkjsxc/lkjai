# kjxlkj Integration

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
  is required in this phase.

## Route Mapping

- `resource.search` -> `GET /api/users/{user}/resources/search`
- `resource.fetch` -> `GET /api/users/{user}/resources/{ref}`
- `resource.history` -> `GET /api/users/{user}/resources/{ref}/history`
- `resource.preview_markdown` -> `POST /api/users/{user}/resources/preview-markdown`
- `resource.create_note` -> `POST /api/users/{user}/resources/notes`
- `resource.create_media` -> `POST /api/users/{user}/resources/media`
- `resource.update_resource` -> `PUT /api/users/{user}/resources/{ref}`
