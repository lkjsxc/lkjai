# Inference Runtime

Owner: `docs/architecture/runtime/inference.md`.
State: canonical inference runtime behavior.

## Goal

Use one real native model engine and surface its health honestly.

## Contract

- The merged native server owns `/api/*` and `/v1/*` routes.
- The server verifies model readiness before reporting chat readiness.
- Model readiness uses the loaded artifact state and `GET /v1/models`.
- Compose process health uses `GET /healthz` so clients can see missing model
  artifacts without a second service dependency.
- `GET /api/model` reports the last known probe result.
- The inference server reports its active device, CUDA availability, GPU name,
  and degradation warning.
- The accepted runtime path is the same path used for quality gates.

## Request And Response

- Request fields: `model`, `messages`, `max_tokens`, `temperature`.
- The runtime consumes `choices[0].message.content` from decoder artifacts.
- Dense and transformer artifacts return unsupported decode, so product chat
  quality gates require decoder exports with the real tokenizer.
- Accepted decoder choices report `cuda_kv_cache` and
  `cuda_contiguous_bf16`, with zero steady-state token allocation.
- Every accepted future model step must return one XML action.
- The runtime system prompt is tracked in native runtime configuration and must
  use the same XML-like serialization as training data.
- The action contract uses `<tool>...`, not `<type>...`.
- Parse repair is allowed in the agent loop, but there is no non-model fallback.
- Plain user text must stay plain when sent to the serving model. Do not wrap
  ordinary chat in a synthetic task envelope unless the prompt is already
  structured that way in the training distribution.

## Failure Semantics

- Request failures surface as transcript `error` events and stop with
  `model_error`.
- Non-success status includes the status code and body text in the transcript.
- Invalid model JSON stops the loop after repair attempts are exhausted.
- CPU diagnostics are allowed only as a visible degraded mode.
- CUDA-unavailable CPU diagnostics must be reported in `/api/model` and the web
  UI.

## Performance Policy

- Prefer CUDA when the native server reports a usable device.
- Target decode keeps state in native-owned buffers during generation.
- Target decode stops as soon as one complete `</action>` is produced.
- Do not use exact prompt lookup, supervised lookup, or canned response tables.

## Defaults

- Direct inference profile:
  `http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=lkjai-scratch-40m`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
