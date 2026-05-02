# Training Configs

## Purpose

Training configs pin long-running scratch model settings.

## Contents

- [scratch_40m_12h.json](scratch_40m_12h.json): active RTX 3070 40M long-run
  estimate.

## Rules

- Keep defaults aligned with the docs canon and native C++/CUDA trainer.
- Training config keys must be implemented by the native loader; unsupported
  keys are rejected rather than silently ignored.
