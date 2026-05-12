# Web Runtime

## Stack

- Native binary named `lkjai-native-server`.
- Merged native HTTP server for `/api/*` and `/v1/*` routing.
- Direct native model-engine calls are the product path.
- `GET /` serves a no-build HTML/CSS/browser-JS chat-first operator page.
- The same process serves `/healthz`, `/api/*`, and `/v1/*`.
- JSON APIs for chat, transcript lists, transcript details, and dense logits.
- OpenAI-compatible model routes for generation.
- Implemented transcripts label user, assistant, and error events.
- Target transcripts also label reasoning, plan, tool call, tool result,
  observation, memory, finish, and confirmation events.
- Client visibility controls are API fields and never alter persisted run
  transcripts.

## Bind Defaults

- Compose binds the host port to `127.0.0.1`.
- The container process listens on `INFERENCE_HOST=0.0.0.0` and
  `INFERENCE_PORT=8080` for the merged server.
- The runtime must not publish a host public network bind by default.

## Model Status

- The header reports model reachability.
- The header reports inference device status.
- The page shows whether the loaded artifact supports chat and keeps dense
  logits in collapsed advanced diagnostics.
- CPU fallback is visible as degraded, not hidden behind a healthy label.
- `/healthz` returns JSON process, artifact, and CUDA capability state from the
  merged server.

## No Node Rule

- Runtime Docker image does not install Node.
- Browser verification does not use Node.
- Frontend behavior is plain HTML, CSS, and browser JavaScript.

## Current Runtime Boundary

- `/api/chat` validates `message`, optional `run_id`, `max_steps`, and
  `visible_event_kinds`.
- `/api/runs` lists JSONL transcripts newest first and clamps `limit` to `100`.
- `/api/dense/status` and `/api/dense/next-token` validate dense demo readiness
  and token-id payloads.
- The runtime persists all events before response filtering.
- Full XML action parsing, tool execution, memory writes, and confirmation
  resume are target work.
