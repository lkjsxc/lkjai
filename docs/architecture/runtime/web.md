# Web Runtime

## Stack

- Rust binary named `lkjai`.
- Axum for HTTP routing.
- Tokio for async runtime.
- Server-rendered HTML for `GET /`.
- JSON APIs for chat and transcripts.
- OpenAI-compatible model endpoint for generation.
- The transcript UI labels reasoning, plan, tool call, tool result, memory,
  finish, assistant, and error events.
- The transcript UI has per-client toggles for every event kind.
- Client visibility controls never alter persisted run transcripts.

## Bind Defaults

- `APP_HOST=127.0.0.1`.
- `APP_PORT=8080`.
- The app must not default to a public network bind.

## Model Status

- The header reports model reachability.
- The header reports inference device status.
- CPU fallback is visible as degraded, not hidden behind a healthy label.

## No Node Rule

- Runtime Docker image does not install Node.
- Browser verification does not use Node.
- Frontend behavior is plain HTML, CSS, and browser JavaScript.
