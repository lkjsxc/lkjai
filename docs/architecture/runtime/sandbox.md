# Sandbox Runtime

Owner: `docs/architecture/runtime/sandbox.md`.
State: canonical sandbox runtime contract.

## Goal

Run the local agent API and read-only file tools in a separate native process
from model inference and static web serving.

## Route Ownership

- `GET /healthz`
- `POST /api/chat`
- `GET /api/model`
- `GET /api/config`
- `GET /api/runs`
- `GET /api/runs/{id}`

The sandbox rejects `/v1/*` and frontend routes. Browser calls may use direct
loopback CORS from `http://127.0.0.1:8080` to `http://127.0.0.1:8082`.

## Model Boundary

- `MODEL_API_URL` defaults to
  `http://inference:8081/v1/chat/completions`.
- `/api/chat` calls the inference service and consumes real generated
  `choices[0].message.content`.
- There is no pretrained fallback, canned assistant text, prompt lookup, or
  deterministic policy-file reply path.
- Model failures are persisted as transcript `error` events and returned with
  `stop_reason=model_error`.

## Mount Policy

- `./data` mounts read-write at `/app/data`.
- `TOOL_WORKSPACE_DIR` defaults to `/app/data/workspace`.
- Compose mounts only these read-only workspace inputs under that directory:
  `docs`, `training`, `corpus`, `configs`, `README.md`, and `compose.yaml`.
- The sandbox must not mount model weights, host `/`, or unrelated host paths.

## Tool API

- Implemented tools: `agent.finish`, `agent.think`, `fs.read`, and `fs.list`.
- `fs.read` and `fs.list` resolve only under `TOOL_WORKSPACE_DIR`.
- Existing read and listing limits remain in force.
- Absolute paths and path traversal that escapes the workspace return tool
  errors instead of reading host files.
- Tool calls, tool results, observations, and errors are persisted to JSONL
  run transcripts.

## Limits And Failures

- `max_steps` is bounded to `[1,64]`.
- Repeated identical non-terminal actions stop as `repeat_action`.
- Invalid XML action responses stop as `invalid_action`.
- Unsupported tools stop as `tool_error`.
- CORS `OPTIONS` preflight returns success for browser direct-port calls.
