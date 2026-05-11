# API Contract

Owner: `docs/product/api.md`.
State: canonical route and payload contract.

This file is the single owner for local product HTTP route shape. Other docs
should link here instead of restating success and failure semantics.

## Routes

- `GET /`: static no-build native browser status/chat page.
- `GET /healthz`: returns JSON process, artifact, and CUDA capability state.
- `POST /api/chat`: runs one bounded agent turn.
- `GET /api/runs/{id}`: returns one run transcript.
- `GET /api/model`: returns model client status including reachability.
- `GET /api/config`: returns local runtime, workspace, and future `kjxlkj`
  adapter status.
- `GET /v1/models`: OpenAI-compatible model readiness route.
- `POST /v1/chat/completions`: OpenAI-compatible model generation route.

`/v1/*` is preserved only for OpenAI-compatible clients. New local APIs should
use unnumbered route names.

## `POST /api/chat` Request

```json
{
  "message": "string",
  "run_id": "optional-string",
  "max_steps": 6,
  "visible_event_kinds": ["user", "assistant", "error"]
}
```

- `visible_event_kinds` is optional.
- When omitted, the server returns all events for API clients.
- When present, the response `events` array contains only matching event kinds.
- Filtering never changes what is persisted to the run transcript.
- Current foundation parity performs one model call. `max_steps` is validated
  and recorded as the future loop bound; full tool-loop stepping is target work.

## `POST /api/chat` Response

```json
{
  "run_id": "string",
  "assistant": "string",
  "events": [],
  "stop_reason": "finish"
}
```

This is the implemented foundation response shape. `assistant` is populated
only when the model endpoint returns OpenAI-compatible `choices` content.
Current native dense and transformer artifacts return unsupported decode from
`/v1/chat/completions`. Decoder artifacts are the product target, but current
decoder choices are partial host-reference usability until accepted CUDA
KV-cache decode evidence exists.

## Decode Capability Matrix

| Artifact kind | `/v1/chat/completions` result | Product role |
|---|---|---|
| `dense` | HTTP `422`, no `choices` | BF16 training and logits diagnostics. |
| `decoder` | `choices`; current host-reference choices report unsupported decode | Product target for accepted chat. |
| `transformer` | HTTP `422`, no `choices` | Reference plumbing only. |

Accepted decoder chat requires `decode_backend=cuda_kv_cache`,
`kv_cache_backend=cuda_contiguous_bf16`, and KV allocation accounting in the
response. Host-reference recompute choices must report
`lkjai_decode_supported=false`.

The exact `/v1/*` route names exist only for OpenAI-compatible clients. Local
runtime routes stay under unnumbered `/api/*` names.

## `GET /api/model` Response

```json
{
  "model": "lkjai-scratch-40m",
  "api_url": "local-native-engine",
  "loaded": true,
  "reachable": true,
  "message": "model server responding",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070",
  "warning": ""
}
```

- `loaded`: a native artifact is loaded.
- `reachable`: the merged native engine is ready to serve model routes.
- `device`: inference device reported by the model engine.
- `cuda_available`: whether the inference server can use CUDA.
- `gpu_name`: CUDA device name when available.
- `warning`: non-empty when serving is degraded, such as CPU fallback.
- `probe_status`: HTTP status from the last `/v1/models` probe.

## `GET /api/config` Response

```json
{
  "service": "lkjai-native-runtime",
  "status": "degraded",
  "degraded": true,
  "degraded_reason": "KJXLKJ_BEARER_TOKEN not configured",
  "bind": {"host": "127.0.0.1", "port": 8080, "local_only": true},
  "workspace_dir": "/app/data/workspace",
  "tool_profile": "readonly",
  "kjxlkj": {
    "api_url": "http://127.0.0.1:8080",
    "user": "default",
    "bearer_token_configured": false,
    "resource_base": "http://127.0.0.1:8080/api/users/default/resources",
    "mutable_tools_enabled": false
  }
}
```

This route is informational. It exposes the API-only `kjxlkj` boundary without
executing resource mutations.

## Event Shape

- `kind`: `user`, `assistant`, `reasoning`, `plan`, `tool_call`,
  `tool_result`, `observation`, `memory_write`, `finish`,
  `confirmation_request`, or `error`.
- `content`: human-readable content.
- `tool`: optional tool name.
- `timestamp`: RFC 3339 timestamp.
- `step`: optional agent loop step.

`reasoning` events come from the model's `<reasoning>` child tag. They are
visible brief rationales and must not contain hidden chain-of-thought detail.

## Error Contract

- Invalid model responses must produce `error` events in `events`.
- If the model server is unreachable, `stop_reason` is `model_error`.
- If the model server responds without assistant content, `stop_reason` is
  `invalid_model_response`.
- If no final assistant action is produced, `stop_reason` must indicate failure.
- If confirmation is required, `stop_reason` is `confirmation_required`.
- `GET /api/model` reflects runtime model client configuration and reachability,
  not benchmarked quality.
- `repeat_action` means the model repeated the same non-terminal action and the
  runtime stopped before wasting more tool calls.

## Verification

```bash
curl -sf http://127.0.0.1:8080/healthz
curl -sf http://127.0.0.1:8080/api/model | jq .
curl -sf -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}' | jq '.stop_reason'
```
