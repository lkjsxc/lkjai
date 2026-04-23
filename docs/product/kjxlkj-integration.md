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
- `resource.update_resource`

## Rules

- `lkjai` should target `kjxlkj` resource APIs, not filesystem-shaped note
  folders.
- Read, search, history, and preview may run directly.
- Create and update operations must first produce
  `{"kind":"request_confirmation", ...}`.
- The mainline integration is API and contract work only. No end-user chat UI
  is required in this phase.
