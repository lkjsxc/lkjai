# Implementation Gaps

Owner: `docs/status/implementation-gaps.md`.
State: current gap map.

## Runtime

- Tool availability must come from one native registry used by prompting,
  dispatch, profile checks, and `/api/config`.
- Profiles are exactly `readonly`, `mutable`, and `disabled`.
- `agent.finish` remains available in `disabled`; other tools are rejected.
- `memory.search` is an implemented read-only JSONL search under
  `data/agent/memory/`.
- `memory.write`, `shell.exec`, `web.fetch`, and `fs.write` remain disabled.
- Resource mutations require `mutable`, a pending confirmation, valid resource
  arguments, and non-empty bearer token.

## Decoder

- Accepted training must fail early when cuDNN SDPA is unavailable.
- Diagnostic training may use reference attention only when it reports
  non-accepted status.
- Accepted reports require train evidence before copied sidecars are promoted.
- Route transcript capture happens after train report validation and is final
  route evidence, not a source for accepted training fields.

## Data

- Tokenizer validation uses `native_xml_tags.hpp` as the atomic tag truth.
- `assistant_masked_sft` accepts exactly one assistant action target per row and
  includes `</action>` in the target span.
- Packed-cache lineage validation covers source, tokenizer, config, checksums,
  sequence length, vocab, and smoke-fixture rejection.

## Web

The browser must disclose artifact kind, CUDA status, stop reason, decode
backend, KV backend, sampler backend, accepted boolean, and direct-model
fallback state when present.
