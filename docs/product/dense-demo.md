# Dense Demo

Owner: `docs/product/dense-demo.md`.
State: canonical dense diagnostic surface.

## Goal

The dense 40M BF16 diagnostics prove that an exported dense artifact can be
loaded and inspected without pretending dense artifacts support chat.

The demo shows:

- model and CUDA readiness,
- token-id input,
- next-token logits,
- sorted top-k entries,
- checksum stability,
- artifact/config metadata,
- optimizer steps, loss, parameter count, and train-report provenance,
- benchmark or diagnostic provenance when present.

## Runtime Boundary

Dense artifacts are not served through sandbox chat routes. The inference
service may load them and report readiness through `/v1/models`, but
`/v1/chat/completions` returns HTTP `422` without `choices`.

Dense logits and top-k checks are native diagnostic helpers and CTest coverage,
not product chat APIs. Do not add `/api/dense/*` routes to the sandbox.

## Request

```json
{
  "tokens": [1, 2, 3],
  "top_k": 8
}
```

- `tokens` is required and must contain integer token ids.
- `top_k` defaults to `8`.
- `top_k` is clamped to the artifact vocab size and rejected when non-positive.
- Text tokenization is target work; this surface intentionally starts with
  token ids so the demo stays native and deterministic.

## Response

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

`decode_supported=false` is required. Dense next-token logits are diagnostic
and demo evidence; they are not autoregressive chat evidence.

## Browser Page

`GET /` serves the static chat-first operator page. It reports model status
through sandbox `/api/model` and direct inference `/v1/models`.

The page must be usable without Node, bundlers, external assets, or generated
frontend code.

## Acceptance

Accepted implementation requires:

- dense runtime helper tests,
- static page contract tests,
- unchanged unsupported dense chat behavior,
- deterministic checksums for smoke artifacts,
- Compose verify passing.

Bounded pilot evidence is enough for this implementation pass. Multi-day
accepted training is a separate operation after these gates pass.
