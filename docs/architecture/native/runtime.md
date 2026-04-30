# Native Server Runtime

## Goal

Serve the scratch model through one native C++/CUDA HTTP process.

## HTTP Contract

- `GET /healthz` reports process and artifact load status.
- `GET /v1/models` reports model readiness, device, CUDA availability, GPU
  name, and warning.
- `POST /v1/chat/completions` accepts `model`, `messages`, `max_tokens`, and
  `temperature`.
- Chat responses keep `choices[0].message.content`.
- Non-success responses include a JSON `error` string.

## Inference Contract

- Load native artifacts from `MODEL_ROOT/MODEL_NAME`.
- Keep decode state and KV cache in native-owned memory.
- Stop generation as soon as one complete `</action>` is produced.
- Do not use supervised lookup, canned responses, or prompt lookup tables.
- CPU execution is allowed only as a visible degraded mode.

## Environment

- `INFERENCE_HOST=0.0.0.0`
- `INFERENCE_PORT=8081`
- `MODEL_ROOT=/models`
- `MODEL_NAME=lkjai-scratch-40m`
