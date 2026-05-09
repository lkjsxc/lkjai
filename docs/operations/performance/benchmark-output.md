# Benchmark Output

Owner: `docs/operations/performance/benchmark-output.md`.
State: canonical index for benchmark output contracts.

Native benchmark tools consume stable train reports and write generated JSON or
CSV summaries for aggregate comparison. This page routes readers to the smaller
field-shape owners instead of duplicating long lists.

## Route By Task

- Stable native train report fields:
  [train-report-fields.md](train-report-fields.md).
- Generated benchmark JSON and CSV artifact shapes:
  [benchmark-artifacts.md](benchmark-artifacts.md).
- Dense diagnostics, accepted-training, and speed-comparison promotion gates:
  [promotion-criteria.md](promotion-criteria.md).

## Status Rules

Successful train reports use top-level `status=success`. Nested checks,
including `logits_check.status` and `reference_check`, continue to use `pass`
or `fail`.

Dense accepted reports set `accepted_cuda_training=true`. Transformer and
partial decoder reports are retained as diagnostics with
`accepted_cuda_training=false` and are excluded from accepted CUDA promotion
aggregates.

Decoder reports may show `decode_backend=host_reference_recompute` and
`kv_cache_backend=none` while still producing route `choices`. That is partial
serving usability, not accepted CUDA KV-cache decode evidence.
