# Web Runtime

## Stack

- Native binary named `lkjai-native-server`.
- Merged native HTTP server for `/api/*` and `/v1/*` routing.
- Direct native model-engine calls are the product path.
- `GET /` returns a compact native service descriptor.
- JSON APIs for chat and transcripts.
- OpenAI-compatible model routes for generation.
- Implemented transcripts label user, assistant, and error events.
- Target transcripts also label reasoning, plan, tool call, tool result,
  observation, memory, finish, and confirmation events.
- Client visibility controls are API fields and never alter persisted run
  transcripts.

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

## Current Runtime Boundary

- `/api/chat` validates `message`, optional `run_id`, `max_steps`, and
  `visible_event_kinds`.
- The runtime persists all events before response filtering.
- Full XML action parsing, tool execution, memory writes, and confirmation
  resume are target work.
