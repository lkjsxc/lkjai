# Training Configs

## Purpose

Training configs pin long-running scratch model settings.

## Contents

- [scratch_40m_12h.json](scratch_40m_12h.json): active RTX 3070 40M long-run
  estimate.
- [profile_20m_3070.json](profile_20m_3070.json): RTX 3070 transformer
  profile target for future native CUDA measurements, not accepted training.
- [profile_120m_5090.json](profile_120m_5090.json): RTX 5090/Blackwell
  transformer profile target, not an acceptance baseline.

## Rules

- Keep defaults aligned with the docs canon and native C++/CUDA trainer.
- Training config keys must be implemented by the native loader; unsupported
  keys are rejected rather than silently ignored.
- Profile training configs must use only `lkjai-train-config-v1` keys and must
  not claim accepted transformer CUDA training.
