# Training Configs

## Purpose

Training configs pin long-running scratch model settings.

## Contents

- [scratch_40m_12h.json](scratch_40m_12h.json): active RTX 3070 40M long-run
  estimate.
- [dense_2h_20m_3070.json](dense_2h_20m_3070.json): calibrated two-hour
  dense BF16 real packed-cache run target for RTX 3070.
- [dense_12h_40m_3070.json](dense_12h_40m_3070.json): dense BF16 40M-size
  long-run target separate from transformer profile configs.
- [decoder_2h_40m_3070.json](decoder_2h_40m_3070.json): tied 40M same-model
  decoder two-hour target for RTX 3070 acceptance.
- [profile_20m_3070.json](profile_20m_3070.json): RTX 3070 transformer
  profile target for future native CUDA measurements, not accepted training.
- [profile_120m_5090.json](profile_120m_5090.json): RTX 5090/Blackwell
  transformer profile target, not an acceptance baseline.

## Rules

- Keep defaults aligned with the docs canon and native C++/CUDA trainer.
- Training config keys must be implemented by the native loader; unsupported
  keys are rejected rather than silently ignored.
- Dense training configs must set `model_kind` to `dense` and point at explicit
  dense-size native configs when parameter count matters.
- Decoder training configs must set `model_kind` to `decoder` and include
  `target_seconds` for wall-clock-bounded acceptance runs.
- Profile training configs must use only `lkjai-train-config` keys and must
  not claim accepted transformer CUDA training.
