# Native Configs

## Purpose

Native dense trainer model configs live here. They are intentionally small JSON
files so the C++ trainer can parse them without a third-party JSON dependency.

## Contents

- [dense_debug_bf16.json](dense_debug_bf16.json): tiny 1-layer verification
  model used by routine native checks.
- [dense_40m_bf16.json](dense_40m_bf16.json): scratch 40M target shape for
  manual smoke runs and production-oriented experiments.

## Rules

- Keep tensor dimensions explicit.
- Routine verification must use the debug config, not the 40M shape.
