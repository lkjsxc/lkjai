# Benchmark Output

Owner: `docs/operations/performance/contracts/benchmark-output.md`.
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

Dense and decoder accepted reports set `accepted_cuda_training=true` only when
their owner contracts pass. Transformer and partial decoder reports are retained
as diagnostics with `accepted_cuda_training=false` and are excluded from
accepted CUDA promotion aggregates.

Accepted decoder benchmark output must include `decode_backend=cuda_kv_cache`,
`kv_cache_backend=cuda_contiguous_bf16`, positive prefill allocation, and zero
steady-state token allocation. Partial decoder output must keep
`accepted_cuda_training=false` when gradients are synthetic or decode uses host
recompute.
