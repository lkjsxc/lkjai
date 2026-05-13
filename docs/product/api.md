# API Contract

Owner: `docs/product/api.md`.
State: canonical route and payload contract.

This file is the single owner for local product HTTP route shape. Other docs
should link here instead of restating success and failure semantics.

## Routes

- Web `GET /`: static no-build chat-first operator page on port `8080`.
- Sandbox `GET /healthz`: returns JSON process state on port `8082`.
- Sandbox `POST /api/chat`: runs one bounded agent turn.
- Sandbox `GET /api/runs`: lists persisted run transcripts.
- Sandbox `GET /api/runs/{id}`: returns one run transcript.
- Sandbox `GET /api/model`: returns model client status including reachability.
- Sandbox `GET /api/config`: returns local runtime, workspace, and future `kjxlkj`
  adapter status.
- Inference `GET /healthz`: returns artifact and CUDA state on port `8081`.
- Inference `GET /v1/models`: OpenAI-compatible model readiness route.
- Inference `POST /v1/chat/completions`: OpenAI-compatible generation route.

`/v1/*` is preserved only for OpenAI-compatible clients. New local APIs use
unnumbered route names. Inference rejects `/api/*`; sandbox rejects `/v1/*`.

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
- The implemented core loop executes `agent.finish`, `agent.think`,
  `fs.list`, and `fs.read`.
- `max_steps` bounds model/action attempts in one user turn.
- Memory, resource, shell, and website tools remain target work.

## `POST /api/chat` Response

```json
{
  "run_id": "string",
  "assistant": "string",
  "events": [],
  "stop_reason": "finish"
}
```

`assistant` is populated only when the model emits an `agent.finish` action
with user-facing content. The sandbox calls the inference service configured by
`MODEL_API_URL`; it does not synthesize assistant text. Current native dense and
transformer artifacts return unsupported decode from `/v1/chat/completions`.
Decoder artifacts are the product target. Decoder choices use the native CUDA
route and disclose accepted CUDA KV-cache decode only when the executed route
evidence is present.

## Direct OpenAI-Compatible Chat

The direct chat path is API-only:

```bash
docker compose --profile inference up --build -d
curl --fail http://127.0.0.1:8081/v1/models
curl -sS -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "decoder-40m-3070",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

For that curl to return `choices`, `.env` must set `MODEL_NAME` to an existing
decoder export such as `decoder-40m-3070`. Dense and transformer artifacts
return HTTP `422` without `choices`; missing artifacts make `GET /v1/models`
return HTTP `503`. This path does not require `GET /` or `/api/chat`.

## Decode Capability Matrix

| Artifact kind | `/v1/chat/completions` result | Product role |
|---|---|---|
| `dense` | HTTP `422`, no `choices` | BF16 training and logits diagnostics. |
| `decoder` | `choices`; accepted or non-accepted CUDA disclosure | Product target for accepted chat. |
| `transformer` | HTTP `422`, no `choices` | Reference plumbing only. |

Accepted decoder chat requires `decode_backend=cuda_kv_cache`,
`kv_cache_backend=cuda_contiguous_bf16`, and KV allocation accounting in the
response. Accepted disclosure also requires the accepted train report copied
beside the artifact and a loaded 40M RTX 3070 decoder shape; a sidecar alone
does not promote the response. Non-accepted decoder route responses report
`lkjai_decode_backend=cuda_reference_kv_cache`,
`lkjai_kv_cache_backend=cuda_contiguous_bf16_partial`,
`lkjai_decode_supported=true`, and `lkjai_decode_accepted=false`.

The exact `/v1/*` route names exist only for OpenAI-compatible clients. Local
runtime routes stay under unnumbered `/api/*` names.

## `GET /api/model` Response

```json
{
  "model": "decoder-40m-3070",
  "api_url": "http://inference:8081/v1/chat/completions",
  "loaded": true,
  "reachable": true,
  "message": "model loaded",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070",
  "warning": "",
  "artifact_kind": "decoder",
  "chat_supported": true,
  "dense_supported": false,
  "tool_profile": "readonly"
}
```

- `loaded`: a native artifact is loaded.
- `reachable`: sandbox can reach the inference service.
- `device`: inference device reported by the model engine.
- `cuda_available`: whether the inference server can use CUDA.
- `gpu_name`: CUDA device name when available.
- `warning`: non-empty when serving is degraded, such as CPU fallback.
- `probe_status`: `200` when the artifact is loaded, otherwise `503`.
- `artifact_kind`: active artifact kind, such as `dense`, `transformer`, or
  `decoder`.
- `chat_supported`: whether `/api/chat` can return decoder assistant content.
- `dense_supported`: whether dense logits diagnostics are available.
- `tool_profile`: active local tool permission profile.

## `GET /api/runs` Response

The optional `limit` query defaults to `20` and is clamped to `100`.

```json
{
  "runs": [
    {
      "run_id": "run-...",
      "created_at": "2026-05-12T00:00:00Z",
      "updated_at": "2026-05-12T00:00:05Z",
      "event_count": 3,
      "last_kind": "assistant",
      "preview": "latest visible run content"
    }
  ]
}
```

## `GET /api/config` Response

```json
{
  "service": "lkjai-native-runtime",
  "status": "degraded",
  "degraded": true,
  "degraded_reason": "KJXLKJ_BEARER_TOKEN not configured",
  "bind": {"host": "127.0.0.1", "port": 8082, "local_only": true},
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

`fs.list` accepts optional `<path>` and defaults to `.`. `fs.read` requires
`<path>`. Both reject absolute paths and workspace escapes, append
`tool_call`, `tool_result`, and `observation` events, and return JSON result
content with `tool`, `status`, `path`, `entries` or `content`, and
`truncated`.

## Error Contract

- Invalid model responses must produce `error` events in `events`.
- If the model server is unreachable, `stop_reason` is `model_error`.
- If the model server responds without assistant content, `stop_reason` is
  `invalid_model_response`.
- If the assistant content is not one valid XML action, `stop_reason` is
  `invalid_action`.
- If no final assistant action is produced, `stop_reason` must indicate failure.
- If the action tool is unavailable in the active profile, `stop_reason` is
  `tool_error`.
- If confirmation is required, `stop_reason` is `confirmation_required`.
- `GET /api/model` reflects runtime model client configuration and reachability,
  not benchmarked quality.
- `repeat_action` means the model repeated the same non-terminal action and the
  runtime stopped before wasting more tool calls.

## Verification

```bash
curl -sf http://127.0.0.1:8082/healthz
curl -sf http://127.0.0.1:8082/api/model | jq .
curl -sf -X POST http://127.0.0.1:8082/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}' | jq '.stop_reason'
```
