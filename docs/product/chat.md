# Chat Product Contract

Owner: `docs/product/chat.md`.
State: canonical product chat surface contract.

## Surface

- `GET /` serves the static no-build browser status/chat page on port `8080`.
- `POST /api/chat` on sandbox port `8082` is the primary product chat surface.
- API clients receive transcript run id, model state, and tool results.
- The implemented foundation persists user, assistant, error, reasoning, plan,
  tool, observation, and finish events.
- Memory and confirmation events are the target agent-loop contract.
- Visibility settings are client-side preferences sent with chat requests.
- The app is local-only by default.
- Browser calls use direct loopback CORS to the sandbox and inference ports.
- There is no login in the current product.

## Current And Target Matrix

| Layer | Current accepted behavior | Target behavior |
|---|---|---|
| Dense artifact | Browser diagnostics, status, logits, top-k, checksum | Stay non-chat and support demo evidence |
| Chat route | Bounded XML-action loop for `agent.finish`, `agent.think`, `fs.list`, and `fs.read` | Memory, summaries, and confirmations |
| Decoder route | CUDA decoder choices with truthful accepted/non-accepted disclosure | Broader streaming and batching metrics |
| Tools | Native read-only filesystem tools | `kjxlkj` resource tools |

## Behavior

- User prompts are sent to sandbox `POST /api/chat`.
- The implemented foundation runs a bounded XML-action loop.
- The first loop executes `agent.finish`, `agent.think`, `fs.list`, and
  `fs.read`.
- Unsupported tools stop as `tool_error` until their contracts land.
- Non-tool prompts use the same HTTP path as later tool prompts.
- Model-status strings are not valid assistant replies.
- The model response must use validated XML actions.
- Simple everyday chat should finish directly with `agent.finish`.
- The runtime must not use canned conversational replies as the default.
- Repeated identical non-terminal model actions stop as `repeat_action`.
- Visible `<reasoning>`, `plan`, `finish`, and `assistant` events are persisted
  when produced by the implemented core agent tools.
- Resource, memory, confirmation, and shell tools remain target work until
  their profile gates pass.
- Every run is persisted as JSONL under `data/agent/runs/`.
- The sandbox must use a real inference endpoint; policy-file dummy responses
  are not an accepted default.
- Current native dense and transformer artifacts do not produce chat responses.
  They return HTTP `422` unsupported decode with no `choices` field.
- Native decoder artifacts are the same-model product target for accepted chat
  responses.
- The web UI must display `/api/chat` attempts even when the model does not
  produce a successful assistant answer. `invalid_action`, `model_error`,
  `invalid_model_response`, and other non-`finish` stop reasons are visible
  chat-attempt outcomes, not blank responses.
- The page shows model kind and decode status from `/api/model` alongside the
  chat stop reason so non-acceptance attempts remain distinguishable from
  accepted chat.
- Decoder choices use native CUDA prefill and contiguous BF16 KV-cache state.
  Accepted artifacts report `lkjai_decode_backend=cuda_kv_cache` and
  `lkjai_kv_cache_backend=cuda_contiguous_bf16`. Artifacts without accepted
  route evidence report `cuda_reference_kv_cache`,
  `cuda_contiguous_bf16_partial`, and `lkjai_decode_accepted=false`.

## Default Safety Boundary

- The default bind host is `127.0.0.1`.
- Host-YOLO is still dangerous because local browser access can trigger
  command execution.
