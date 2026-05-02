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
- Transformer artifacts load through `/v1/models`; autoregressive chat decode
  returns an explicit unsupported-decode error until the decode slice lands.
- `lkjai-native-logits-check` is the accepted inference proof for this slice.
- Do not use supervised lookup, canned responses, or prompt lookup tables.
- CPU execution is allowed only as a visible degraded mode.
- The smoke artifact is a transformer artifact with named tensors. It is a
  numerics and artifact gate, not a behavioral competency artifact.

## Environment

- `INFERENCE_HOST=0.0.0.0`
- `INFERENCE_PORT=8081`
- `MODEL_ROOT=/models`
- `MODEL_NAME=lkjai-scratch-40m`
