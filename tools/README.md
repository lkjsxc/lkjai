# Tools

## Purpose

Support tooling lives here. These are not production runtime entry points.

## Contents

- [benchmarks/](benchmarks/): Docker-backed training benchmark helpers.
- [diagnostics/](diagnostics/): local diagnostic collection utilities.
- [experiments/](experiments/): experimental scripts.
- [kimi-corpus/](kimi-corpus/): Kimi CLI synthetic corpus generator.
- [packed-reader/](packed-reader/): Rust packed-token reader utility.
- [reports/](reports/): report generation helpers.

## Rules

- Keep experimental state out of tracked files.
- Prefer deterministic CLI output that agents can parse.
