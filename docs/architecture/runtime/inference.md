# Model Client Runtime

## Goal

Call one real OpenAI-compatible inference endpoint and surface its health
honestly.

## Contract

- The web runtime must call a separate inference endpoint.
- The runtime must verify inference health before serving chat requests.
- Health probe uses `GET /v1/models` with a `5` second timeout.
- `GET /api/model` reports the last known probe result.
- The accepted runtime path is the same path used for quality gates.

## Request And Response

- Request fields: `model`, `messages`, `max_tokens`, `temperature`.
- The Rust client consumes `choices[0].message.content`.
- Every model step must return one strict JSON action.
- Parse repair is allowed in the agent loop, but there is no non-model fallback.

## Failure Semantics

- Request failures surface as transcript `error` events and stop with
  `model_error`.
- Non-success status includes the status code and body text in the transcript.
- Invalid model JSON stops the loop after repair attempts are exhausted.

## Defaults

- `MODEL_API_URL=http://127.0.0.1:8081/v1/chat/completions`
- `MODEL_NAME=lkjai-scratch-60m`
- `MODEL_MAX_NEW_TOKENS=512`
- `MODEL_TEMPERATURE=0.2`
