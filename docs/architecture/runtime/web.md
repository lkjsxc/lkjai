# Web Runtime

## Stack

- Rust binary named `lkjai`.
- Axum for HTTP routing.
- Tokio for async runtime.
- Server-rendered HTML for `GET /`.
- JSON APIs for chat and transcripts.
- OpenAI-compatible model endpoint for generation.

## Bind Defaults

- `APP_HOST=127.0.0.1`.
- `APP_PORT=8080`.
- The app must not default to a public network bind.

## No Node Rule

- Runtime Docker image does not install Node.
- Browser verification does not use Node.
- Frontend behavior is plain HTML, CSS, and browser JavaScript.
