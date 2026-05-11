# Training Configs

## Purpose

Training configs pin long-running scratch model settings.

## Contents

- [dense_40m_accepted_3070.json](dense_40m_accepted_3070.json): active RTX
  3070 dense 40M accepted browser-demo training configuration.
- [dense_2h_20m_3070.json](dense_2h_20m_3070.json): calibrated two-hour
  dense BF16 real packed-cache run target for RTX 3070.
- [scratch_40m_12h.json](scratch_40m_12h.json): legacy scratch 40M manual
  run estimate. Do not use it as a Compose default.
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
