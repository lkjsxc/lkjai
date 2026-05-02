# Benchmarking Contract

## Required Metrics

Every performance run records:

- commit SHA,
- Docker image tag,
- GPU name and compute capability,
- driver, CUDA, cuDNN, and native build versions,
- training preset and JSON config path,
- batch size, gradient accumulation, launch mode, current trainer mode, and
  `model_kind`,
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
- Bounded `lkjai-native-train --train --mode transformer` cases only when the
  emitted `train-report.json` declares `model_kind=transformer` and includes
  transformer timing and checksum fields.
- Batch size and gradient accumulation values that the native trainer reports.

Benchmark tooling consumes `DATA_DIR/runs/train-report.json` or the stdout JSON
with the same schema. It aggregates transformer-specific timing and checksum
fields only from emitted reports. It does not require or parse
`perf-steps.jsonl`.

Attention backend, FP16/AMP, activation checkpoint, and CUDA Graph sweeps are
roadmap benchmarks after those switches are implemented in the native trainer.

## Full-Run Rule

After bounded benchmarks, run a fresh full pipeline in a new data directory.
Record the final training summary, eval outputs, and selected benchmark case in
[../training/iteration.md](../training/iteration.md).
