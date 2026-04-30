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
- Read, search, history, and preview may run directly.
- Create and update operations must first produce
  `{"kind":"request_confirmation", ...}` and must not execute until the next
  user turn explicitly confirms the pending operation.
- The mainline integration is API and contract work only. No end-user chat UI
  is required in this phase.

## Route Mapping

- `resource.search` -> `GET /api/resources/search`
- `resource.fetch` -> `GET /api/resources/{id}`
- `resource.history` -> `GET /api/resources/{id}/history`
- `resource.preview_markdown` -> `POST /admin/markdown-preview`
- `resource.create_note` -> `POST /api/resources/notes`
- `resource.create_media` -> `POST /api/resources/media`
- `resource.update_resource` -> `PUT /api/resources/{id}`
