# Training Configs

Owner: `configs/training/README.md`.
State: canonical documentation.


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
- [decoder_40m_agent_sft_3070.json](decoder_40m_agent_sft_3070.json): 40M
  decoder `assistant_masked_sft` target for XML-action traces.
- [decoder_4h_chat_attempt_3070.json](decoder_4h_chat_attempt_3070.json):
  four-hour non-acceptance `assistant_masked_sft` chat-attempt run that exports
  over the default `decoder-40m-3070` serving name. It uses `128` token
  sequences and frequent checkpoints to keep the attempt lane runnable on
  RTX 3070.
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
  `target_seconds` for wall-clock-bounded runs. Only the committed
  `decoder_2h_40m_3070.json` config is the acceptance lane.
- Profile training configs must use only `lkjai-train-config` keys and must
  not claim accepted transformer CUDA training.
