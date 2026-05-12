# API Contract

Owner: `docs/product/api.md`.
State: canonical route and payload contract.

This file is the single owner for local product HTTP route shape. Other docs
should link here instead of restating success and failure semantics.

## Routes

- `GET /`: static no-build chat-first operator page.
- `GET /healthz`: returns JSON process, artifact, and CUDA capability state.
- `GET /api/dense/status`: returns dense demo readiness and artifact metadata.
- `POST /api/dense/next-token`: returns dense next-token top-k logits summary.
- `POST /api/chat`: runs one bounded agent turn.
- `GET /api/runs`: lists persisted run transcripts.
- `GET /api/runs/{id}`: returns one run transcript.
- `GET /api/model`: returns model client status including reachability.
- `GET /api/config`: returns local runtime, workspace, and future `kjxlkj`
  adapter status.
- `GET /v1/models`: OpenAI-compatible model readiness route.
- `POST /v1/chat/completions`: OpenAI-compatible model generation route.

`/v1/*` is preserved only for OpenAI-compatible clients. New local APIs use
unnumbered route names.

## Dense Demo Request

```json
{
  "tokens": [1, 2, 3],
  "top_k": 8
}
```

`tokens` is a required array of token ids. `top_k` is optional and defaults to
`8`.

## Dense Demo Response

```json
{
  "status": "success",
  "model_kind": "dense",
  "decode_supported": false,
  "checksum": "string",
  "top_token": 42,
  "top_k": [
    {"id": 42, "logit": 1.25, "prob": 0.33}
  ]
}
```

Dense demo routes do not return chat `choices`. They expose logits evidence for
the loaded artifact.

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
- The implemented core loop executes `agent.finish` and `agent.think`.
- `max_steps` bounds model/action attempts in one user turn.
- Filesystem, memory, resource, shell, and website tools remain target work.

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
with user-facing content. Current native dense and transformer artifacts return
unsupported decode from `/v1/chat/completions`. Decoder artifacts are the
product target. Decoder choices use the native CUDA route and disclose accepted
CUDA KV-cache decode only when the executed route evidence is present.

## Decode Capability Matrix

| Artifact kind | `/v1/chat/completions` result | Product role |
|---|---|---|
| `dense` | HTTP `422`, no `choices` | BF16 training and logits diagnostics. |
| `decoder` | `choices`; accepted or non-accepted CUDA disclosure | Product target for accepted chat. |
| `transformer` | HTTP `422`, no `choices` | Reference plumbing only. |

Accepted decoder chat requires `decode_backend=cuda_kv_cache`,
`kv_cache_backend=cuda_contiguous_bf16`, and KV allocation accounting in the
response. Non-accepted decoder route responses report
`lkjai_decode_backend=cuda_reference_kv_cache`,
`lkjai_kv_cache_backend=cuda_contiguous_bf16_partial`,
`lkjai_decode_supported=true`, and `lkjai_decode_accepted=false`.

The exact `/v1/*` route names exist only for OpenAI-compatible clients. Local
runtime routes stay under unnumbered `/api/*` names.

## `GET /api/model` Response

```json
{
  "model": "lkjai-scratch-40m",
  "api_url": "local-native-engine",
  "loaded": true,
  "reachable": true,
  "message": "model loaded",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3070",
  "warning": "",
  "artifact_kind": "dense",
  "chat_supported": false,
  "dense_supported": true,
  "tool_profile": "readonly"
}
```

- `loaded`: a native artifact is loaded.
- `reachable`: the merged native engine is ready to serve model routes.
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
curl -sf http://127.0.0.1:8080/healthz
curl -sf http://127.0.0.1:8080/api/model | jq .
curl -sf -X POST http://127.0.0.1:8080/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hello"}' | jq '.stop_reason'
```
