# Chat Product Contract

## Surface

- `GET /` serves the first-screen chat application.
- The UI exposes a prompt box, run transcript, and tool results.
- The app is local-only by default.
- There is no login in v1.

## Behavior

- User prompts are sent to `POST /api/chat`.
- The model response may request deterministic tool calls.
- Tool calls and outputs are displayed in the transcript.
- Every run is persisted as JSONL under `data/agent/runs/`.

## Default Safety Boundary

- The default bind host is `127.0.0.1`.
- Host-YOLO is still dangerous because local browser access can trigger
  command execution.
