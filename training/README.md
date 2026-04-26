# Training

## Purpose

Training owns the Python package, tests, and packaging metadata for scratch
model preparation, tokenizer training, model training, evaluation, and manifest
export.

## Contents

- [package/](package/): import root containing `lkjai_train`.
- [tests/](tests/): pytest suite and fixtures.
- [pyproject.toml](pyproject.toml): Python project metadata and pytest config.

## Rules

- Import modules with `PYTHONPATH=training/package`.
- Long jobs run through Docker Compose.
- Generated artifacts belong under `data/`, not in this directory.
