# Native Server Runtime

Owner: `docs/architecture/native/contracts/runtime.md`.
State: canonical native HTTP route and readiness contract.

## Goal

Serve the scratch model and local agent API through split native inference and
sandbox HTTP processes.

## HTTP Contract

- Inference `GET /healthz` returns artifact load and CUDA capability state.
- Static web `GET /` serves the no-build browser status/chat page.
- Inference `GET /v1/models` reports model readiness, device, CUDA availability, GPU
  name, hardware/build capability fields, and warning.
- Inference `POST /v1/chat/completions` accepts `model`, `messages`,
  `max_tokens`, and `temperature`.
- Sandbox `POST /api/chat`, `GET /api/model`, `GET /api/config`, and
  `GET /api/runs/{id}` are runtime routes in a separate process.
- Successful decoder chat responses keep `choices[0].message.content`.
- Non-success responses include a JSON `error` string.
- Capability fields follow [capability.md](../overview/capability.md).
- Inference rejects `/api/*` and frontend routes. Sandbox rejects `/v1/*` and
  frontend routes. Both native services allow CORS preflight.

## Inference Contract

- Load native artifacts from `MODEL_ROOT/MODEL_NAME`.
- Dense and transformer artifacts load through `/v1/models`; autoregressive chat
  decode returns HTTP `422` with an explicit unsupported-decode error and no
  `choices` field for those kinds.
- Decoder artifacts are the only artifacts that may return
  `/v1/chat/completions` choices. Current CUDA reference choices are partial
  usability and must disclose non-accepted decode.
- The sandbox posts to the inference service through `MODEL_API_URL`.
- `lkjai-native-logits-check` is the accepted inference proof for this slice.
- Do not use supervised lookup, canned responses, or prompt lookup tables.
- CPU execution is allowed only as a visible degraded mode outside dense CUDA
  training.
- The smoke artifact is a dense artifact with named tensors. It is a numerics
  and artifact gate, not a behavioral competency artifact.

## Decode Target

The accepted decoder decode slice must provide:

- prefill from prompt tokens,
- contiguous KV cache for the first implementation,
- zero steady-state device allocations per generated token,
- on-device temperature, top-k/top-p, and argmax or multinomial sampling,
- stop-token and `</action>` completion detection,
- paged KV cache only after continuous batching is introduced.

The current non-accepted decoder runtime reports
`lkjai_decode_backend=cuda_reference_kv_cache`,
`lkjai_kv_cache_backend=cuda_contiguous_bf16_partial`,
`lkjai_decode_supported=true`, and `lkjai_decode_accepted=false`.

Accepted runtime disclosure must not be driven by `decoder_acceptance.json`
alone. It requires real CUDA KV-cache decode, an adjacent accepted train
report, loaded 40M RTX 3070 decoder shape, and route evidence.

## Environment

- `INFERENCE_HOST=0.0.0.0`
- `INFERENCE_PORT=8081`
- `MODEL_ROOT=/models`
- `MODEL_NAME=decoder-40m-3070`
- `KJXLKJ_API_URL=http://127.0.0.1:8080`
- `KJXLKJ_USER=default`
- `KJXLKJ_BEARER_TOKEN=` leaves `/api/config` visibly degraded.
