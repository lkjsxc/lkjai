# Benchmarks

## Purpose

Benchmark helpers run bounded training experiments and collect comparable
metrics.

## Contents

- [benchmark_reports.py](benchmark_reports.py): train-report summary and
  promotion-gating helpers.
- [dense_40m_compat_support.py](dense_40m_compat_support.py): bounded 40M
  compatibility runner internals.
- [dense_debug_runner.py](dense_debug_runner.py): dense debug promotion runner
  internals.
- [dense_debug_support.py](dense_debug_support.py): dense debug Docker and
  artifact helpers.
- [promote_dense_debug.py](promote_dense_debug.py): accepted dense debug
  promotion entrypoint.
- [run_matrix.py](run_matrix.py): benchmark matrix launcher.
- [run_dense_40m_compat.py](run_dense_40m_compat.py): bounded
  `native_40m_bf16` compatibility start check.
- [run_support.py](run_support.py): shared Docker and metrics helpers.

## Rules

- Write outputs under ignored artifact directories.
- Build training containers from `ops/docker/Dockerfile.native`.
- Use only supported native modes: dense smoke and bounded packed-cache train.
- Parse `runs/train-report.json` or stdout report JSON; do not depend on
  `perf-steps.jsonl`.
