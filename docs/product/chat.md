# Chat Product Contract

## Surface

- `GET /` serves the first-screen chat application.
- The UI exposes a chat transcript, prompt box, run id, model state, and tool
  results.
- The UI displays plan, tool, observation, memory, assistant, and error events.
- The app is local-only by default.
- There is no login in v1.

## Behavior

- User prompts are sent to `POST /api/chat`.
- The agent may run several model/tool steps before answering.
- Non-tool prompts are answered through the same agent loop.
- Model-status strings are not valid assistant replies.
- The model response must use validated JSON actions.
- Tool calls and outputs are displayed in the transcript.
- Memory writes are displayed in the transcript.
- Every run is persisted as JSONL under `data/agent/runs/`.
- The runtime must use a real model endpoint; policy-file dummy responses are not
  an accepted default.

## Default Safety Boundary

- The default bind host is `127.0.0.1`.
- Host-YOLO is still dangerous because local browser access can trigger
  command execution.
