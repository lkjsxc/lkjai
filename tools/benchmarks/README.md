# Benchmarks

## Purpose

Benchmark helpers run bounded training experiments and collect comparable
metrics.

## Contents

- [accepted_training_reports.py](accepted_training_reports.py): accepted dense
  training report validation and Markdown summary helpers.
- [benchmark_reports.py](benchmark_reports.py): train-report summary and
  promotion-gating helpers.
- [dense_40m_compat_support.py](dense_40m_compat_support.py): bounded 40M
  compatibility runner internals.
- [dense_accepted_training_support.py](dense_accepted_training_support.py):
  accepted dense real packed-cache runner internals.
- [dense_accepted_training_cache.py](dense_accepted_training_cache.py):
  accepted dense packed-cache build/validate helpers.
- [dense_accepted_training_io.py](dense_accepted_training_io.py): accepted
  dense paths, Docker command, JSON log, and payload helpers.
- [dense_accepted_training_train.py](dense_accepted_training_train.py):
  accepted dense train/check command helpers.
- [dense_debug_runner.py](dense_debug_runner.py): dense debug promotion runner
  internals.
- [dense_debug_support.py](dense_debug_support.py): dense debug Docker and
  artifact helpers.
- [dense_learning_control_io.py](dense_learning_control_io.py): synthetic dense
  learning-control cache and Docker command helpers.
- [dense_learning_control_support.py](dense_learning_control_support.py):
  dense learning-control validation and summary helpers.
- [promote_dense_debug.py](promote_dense_debug.py): accepted dense debug
  promotion entrypoint.
- [run_matrix.py](run_matrix.py): benchmark matrix launcher.
- [run_dense_40m_compat.py](run_dense_40m_compat.py): bounded
  `native_40m_bf16` compatibility start check.
- [run_dense_accepted_training.py](run_dense_accepted_training.py): accepted
  dense real packed-cache training runner.
- [run_dense_learning_control.py](run_dense_learning_control.py): synthetic
  dense learning-control proof runner.
- [run_support.py](run_support.py): shared Docker and metrics helpers.

## Rules

- Write outputs under ignored artifact directories.
- Build training containers from `ops/docker/Dockerfile.native`.
- Use only supported native modes: dense smoke and bounded packed-cache train.
- Parse `runs/train-report.json` or stdout report JSON; do not depend on
  `perf-steps.jsonl`.
