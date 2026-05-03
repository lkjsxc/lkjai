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
- dense loader backend, row layout, matmul plan cache flag, buffer reuse flag,
  and timing source.

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
`model_kind=dense`, `accepted_cuda_training=true`, `implementation_status=accepted`,
`status=success`, `loader_backend=persistent_packed_cache_reader`,
`matmul_plan_cache_enabled=true`, `buffer_reuse_enabled=true`, decreasing loss,
artifact checksums, positive throughput, non-negative CUDA-event timings, and
passing logits/reference tolerance checks.
Reports with `run_purpose=bounded_compatibility_start_check` are listed as
compatibility diagnostics and are rejected by promotion aggregates even when
loss decreases.
Diagnostic summaries may still list experimental transformer timings and
checksums. The tooling does not require or parse `perf-steps.jsonl`.

RTX 3070 8GB remains the acceptance gate. RTX 5090/Blackwell results are useful
benchmark profiles, but they do not promote a run unless the RTX 3070 gate also
passes. See [hardware-profiles.md](hardware-profiles.md).

## Accepted Dense Debug Promotion

The accepted CUDA promotion for this batch is a debug-shape correctness and
artifact contract run only. It uses `native_debug_bf16`, the checked-in
seq16/vocab256 packed cache, batch size 1, gradient accumulation 1, and 128
optimizer steps. It is not a 40M or production-scale performance baseline.

- run id: `dense-debug-promote-20260502-175250`
- command: `python3 tools/benchmarks/promote_dense_debug.py --run-id dense-debug-promote-20260502-175250 --steps 128 --resume-steps 1 --sample-interval 0.25`
- device: NVIDIA GeForce RTX 3070
- backend: forward `cuda_bf16_cublaslt`, backward
  `cuda_bf16_cublaslt_scatter`,
  optimizer `cuda_adamw_fp32`
- shape: batch 1, seq_len 16, hidden 32, vocab 256, parameters 16,384
- loss: 5.54545 initial to 5.21614 final
- throughput: 10,985.3 tokens/sec over 0.186431 trainer seconds
- H2D fraction: 0.040913
- phase fractions: batch_load 0.065566, h2d 0.040913, forward 0.602024,
  backward 0.059562, optimizer 0.051176, checkpoint 0.058160, export 0.013001
- checksums: checkpoint/export `d24dcbe4136d6480`, logits `e8cc855981c7832f`
- logits reference check: `pass`, max_abs_diff 0.00031428, tolerance 0.01
- resume check: `success`, start_step 128, optimizer_steps 129

Attention backend, FP16/AMP, activation checkpoint, and CUDA Graph sweeps are
roadmap benchmarks after those switches are implemented in the native trainer.

## Larger Dense Compatibility

The 40M dense shape is checked with a bounded start-only command:

```sh
python3 tools/benchmarks/run_dense_40m_compat.py \
  --run-id RUN_ID \
  --steps 4 \
  --sequence-count 8 \
  --sample-interval 0.25
```

The runner builds and validates a deterministic cache under
`data/perf-runs/<run-id>/packed/`, then runs dense CUDA with
`native_40m_bf16`, `seq_len=1024`, batch size 1, gradient accumulation 1,
four optimizer steps, checkpoint interval 4, and learning rate `0.0003`.
Outputs are written to
`artifacts/benchmarks/<run-id>/dense_40m_compat_4/repeat-01/`. This is
compatibility-only until a later 40M run satisfies the promotion criteria.

## Full-Run Rule

After bounded benchmarks, run a fresh full pipeline in a new data directory.
Record the final training summary, eval outputs, and selected benchmark case in
[../training/iteration.md](../training/iteration.md).
