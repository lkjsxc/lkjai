# Chat Product Contract

## Surface

- `GET /` serves a compact native service descriptor.
- `POST /api/chat` is the primary product chat surface.
- API clients receive transcript run id, model state, and tool results.
- The implemented foundation persists user, assistant, and error events.
- Reasoning, plan, tool, observation, memory, finish, and confirmation events
  are the target agent-loop contract.
- Visibility settings are client-side preferences sent with chat requests.
- The app is local-only by default.
- There is no login in v1.

## Behavior

- User prompts are sent to `POST /api/chat`.
- The implemented foundation makes one model call per request.
- The target agent loop may run several model/tool steps before answering.
- Non-tool prompts use the same HTTP path as later tool prompts.
- Model-status strings are not valid assistant replies.
- The model response must use validated XML actions.
- Simple everyday chat should finish directly with `agent.finish`.
- The runtime must not use canned conversational replies as the default.
- Repeated identical non-terminal model actions stop as `repeat_action`.
- Tool calls, outputs, visible `<reasoning>`, and memory writes are target
  transcript event kinds until tool execution lands.
- Every run is persisted as JSONL under `data/agent/runs/`.
- The runtime must use a real model endpoint; policy-file dummy responses are not
  an accepted default.
- Current native dense and transformer artifacts do not produce chat responses.
  They return HTTP `422` unsupported decode with no `choices` field.
- Native decoder artifacts are the same-model product target for successful
  chat `choices` responses.

## Default Safety Boundary

- The default bind host is `127.0.0.1`.
- Host-YOLO is still dangerous because local browser access can trigger
  command execution.
