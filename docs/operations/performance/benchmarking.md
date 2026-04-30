# Benchmarking Contract

## Required Metrics

Every performance run records:

- commit SHA,
- Docker image tag,
- GPU name and compute capability,
- driver, CUDA, cuDNN, and native build versions,
- training preset and JSON config path,
- batch size, gradient accumulation, AMP, compile mode, attention backend,
- median and p95 microstep seconds,
- median input tokens/sec,
- loader wait, H2D, forward, backward, and optimizer timing.

## Required Artifacts

Write generated benchmark outputs under ignored `artifacts/` paths:

- `artifacts/diagnostics/<run-id>/summary.json`
- `artifacts/benchmarks/<run-id>/summary.csv`
- `artifacts/benchmarks/<run-id>/aggregate.json`
- `artifacts/reports/<run-id>/training-performance-report.md`
- optional profiler traces under `artifacts/profiles/<run-id>/`

Tracked docs may summarize curated results, but generated reports do not live
outside `artifacts/`.

## Benchmark Matrix

The bounded matrix must include at least:

- `sdpa`, `sdpa_flash`, and `auto` attention backends,
- BF16, FP16, and AMP off,
- compile off, `reduce-overhead`, and `max-autotune-no-cudagraphs`,
- activation checkpoint off and every-n,
- batch sizes that fit the active GPU,
- legacy, mapped, and batch-mapped data loading.

## Full-Run Rule

After bounded benchmarks, run a fresh full pipeline in a new data directory.
Record the final training summary, eval outputs, and selected benchmark case in
[../training/iteration.md](../training/iteration.md).
