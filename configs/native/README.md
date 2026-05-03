# Native Configs

## Purpose

Native model-shape configs live here. They are intentionally small JSON files
so the C++ trainer can parse them without a third-party JSON dependency.

## Contents

- [native_debug_bf16.json](native_debug_bf16.json): tiny verification shape
  used by routine native dense checks.
- [native_accepted_dense_bf16.json](native_accepted_dense_bf16.json): small
  real packed-cache dense target for the `accepted_training` proof run.
- [native_transformer_debug_bf16.json](native_transformer_debug_bf16.json):
  tiny verification shape for explicit native transformer training. It uses
  learned positional embeddings and untied embeddings.
- [native_20m_bf16_3070.json](native_20m_bf16_3070.json): RTX 3070
  transformer profile target for future CUDA-forward profiling. It is not
  accepted transformer CUDA training.
- [native_40m_bf16.json](native_40m_bf16.json): scratch 40M target shape for
  manual smoke runs and production-oriented experiments. It is not the current
  accepted-training target.
- [native_120m_bf16_5090.json](native_120m_bf16_5090.json): RTX
  5090/Blackwell transformer profile target. It is not an acceptance baseline.

## Rules

- Keep tensor dimensions explicit.
- Routine verification must use the debug config, not the 40M shape.
- Profile configs must satisfy `heads * head_dim == hidden_size` and
  `heads % kv_heads == 0`, but they do not promote transformer training.
