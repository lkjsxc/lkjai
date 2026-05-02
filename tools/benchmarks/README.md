# Benchmarks

## Purpose

Benchmark helpers run bounded training experiments and collect comparable
metrics.

## Contents

- [run_matrix.py](run_matrix.py): benchmark matrix launcher.
- [run_support.py](run_support.py): shared Docker and metrics helpers.

## Rules

- Write outputs under ignored artifact directories.
- Build training containers from `ops/docker/Dockerfile.native`.
- Use only supported native modes: dense smoke and bounded packed-cache train.
- Parse `runs/train-report.json` or stdout report JSON; do not depend on
  `perf-steps.jsonl`.
