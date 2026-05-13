# Web Runtime

## Stack

- Static files under `web/`.
- Static image built by `Dockerfile.web`; no Node, model mounts, data mounts,
  or GPU access.
- `GET /` serves the no-build HTML/CSS/browser-JS operator page.
- Browser JavaScript calls `http://127.0.0.1:8082/api/*` for sandbox APIs.
- Browser JavaScript calls `http://127.0.0.1:8081/v1/*` only for direct model
  diagnostics.
- Implemented transcripts label user, assistant, error, reasoning, plan, tool
  call, tool result, observation, and finish events.
- Target transcripts also label memory and confirmation events.
- Client visibility controls are API fields and never alter persisted run
  transcripts.

## Bind Defaults

- Compose binds the host port to `127.0.0.1`.
- The container process listens on static HTTP port `8080`.
- The runtime must not publish a host public network bind by default.

## Model Status

- The header reports model reachability.
- The header reports inference device status.
- The page shows whether the loaded artifact supports chat and keeps dense
  logits in collapsed advanced diagnostics.
- CPU fallback is visible as degraded, not hidden behind a healthy label.
- Model and CUDA status are read from sandbox `/api/model`, which probes the
  inference service.

## No Node Rule

- Web Docker image does not install Node.
- Browser verification does not use Node.
- Frontend behavior is plain HTML, CSS, and browser JavaScript.

## Current Runtime Boundary

- The web process does not own product API routes.
- Sandbox `/api/chat` validates `message`, optional `run_id`, `max_steps`, and
  `visible_event_kinds`.
- Sandbox `/api/runs` lists JSONL transcripts newest first and clamps `limit`
  to `100`.
- XML action parsing and native read-only filesystem tool execution live in the
  sandbox process. Memory writes and confirmation resume are target work.
