# Native Configs

## Purpose

Native model-shape configs live here. They are intentionally small JSON files
so the C++ trainer can parse them without a third-party JSON dependency.

## Contents

- [native_debug_bf16.json](native_debug_bf16.json): tiny verification shape
  used by routine native dense checks.
- [native_40m_bf16.json](native_40m_bf16.json): scratch 40M target shape for
  manual smoke runs and production-oriented experiments.

## Rules

- Keep tensor dimensions explicit.
- Routine verification must use the debug config, not the 40M shape.
