# Model Client Runtime

## Goal

Call one real OpenAI-compatible inference endpoint and surface its health
honestly.

## Contract

- The web runtime must call a separate inference endpoint.
- The runtime must verify inference health before serving chat requests.
- Health probe uses `GET /v1/models` with a `5` second timeout.
- `GET /api/model` reports the last known probe result.
- The inference server reports its active device, CUDA availability, GPU name,
  and degradation warning.
- The accepted runtime path is the same path used for quality gates.

## Request And Response

- Request fields: `model`, `messages`, `max_tokens`, `temperature`.
- The Rust client consumes `choices[0].message.content`.
- Every model step must return one XML action.
- Parse repair is allowed in the agent loop, but there is no non-model fallback.
- Plain user text must stay plain when sent to the serving model. Do not wrap
  ordinary chat in a synthetic task envelope unless the prompt is already
  structured that way in the training distribution.

## Failure Semantics

- Request failures surface as transcript `error` events and stop with
  `model_error`.
- Non-success status includes the status code and body text in the transcript.
- Invalid model JSON stops the loop after repair attempts are exhausted.
- CPU inference is allowed only as a visible degraded mode.
- CUDA-unavailable CPU fallback must be reported in `/api/model` and the web UI.

## Performance Policy

- Prefer CUDA when `torch.cuda.is_available()` is true.
- Use `torch.inference_mode()` during generation.
- Stop generation as soon as one complete `</action>` is produced.
- Do not use exact prompt lookup, supervised lookup, or canned response tables.

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=lkjai-scratch-60m`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
