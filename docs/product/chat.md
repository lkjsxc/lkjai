# Chat Product Contract

## Surface

- `GET /` serves the first-screen chat application.
- The UI exposes a chat transcript, prompt box, run id, model state, and tool
  results.
- The app is local-only by default.
- There is no login in v1.

## Behavior

- User prompts are sent to `POST /api/chat`.
- Non-tool prompts are answered by real model generation from the loaded export.
- Model-status strings are not valid assistant replies.
- The model response may request deterministic tool calls.
- Tool calls and outputs are displayed in the transcript.
- Natural-language tool requests are accepted for shell commands, URL fetches,
  file reads, file writes, and directory listings.
- Every run is persisted as JSONL under `data/agent/runs/`.

## Default Safety Boundary

- The default bind host is `127.0.0.1`.
- Host-YOLO is still dangerous because local browser access can trigger
  command execution.
