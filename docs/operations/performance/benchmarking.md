# Benchmarking Contract

## Required Metrics

Every performance run records:

- commit SHA,
- Docker image tag,
- GPU name and compute capability,
- driver, CUDA, cuDNN, and native build versions,
- training preset and JSON config path,
- batch size, gradient accumulation, launch mode, current trainer mode,
  `model_kind`, `accepted_cuda_training`, and `implementation_status`,
- median optimizer-step seconds from `train-report.json`,
- median input tokens/sec,
- loader wait, H2D, forward, backward, and optimizer timing.

## Required Artifacts

Write generated benchmark outputs under ignored `artifacts/` paths:

- `artifacts/diagnostics/<run-id>/summary.json`
- `artifacts/benchmarks/<run-id>/summary.csv`
- `artifacts/benchmarks/<run-id>/aggregate.json`
- `artifacts/benchmarks/<run-id>/<case>/repeat-NN/train-report.json`
- `artifacts/reports/<run-id>/training-performance-report.md`
- optional profiler traces under `artifacts/profiles/<run-id>/`

Tracked docs may summarize curated results, but generated reports do not live
outside `artifacts/`.

## Benchmark Matrix

The current bounded matrix uses supported native trainer modes only:

- `lkjai-native-train --smoke --steps N` for reproducible dense CUDA smoke.
- Bounded `lkjai-native-train --train` packed-cache cases when a matching
  packed cache is present.
- Bounded `lkjai-native-train --train --mode transformer` cases may be kept as
  experimental diagnostics only. They must emit `accepted_cuda_training=false`
  and are excluded from accepted CUDA promotion aggregates.
- Batch size and gradient accumulation values that the native trainer reports.

Benchmark tooling consumes `DATA_DIR/runs/train-report.json` or the stdout JSON
with the same schema. Promotion aggregates use only reports with
`accepted_cuda_training=true`; diagnostic summaries may still list experimental
transformer timings and checksums. The tooling does not require or parse
`perf-steps.jsonl`.

Attention backend, FP16/AMP, activation checkpoint, and CUDA Graph sweeps are
roadmap benchmarks after those switches are implemented in the native trainer.

## Full-Run Rule

After bounded benchmarks, run a fresh full pipeline in a new data directory.
Record the final training summary, eval outputs, and selected benchmark case in
[../training/iteration.md](../training/iteration.md).
