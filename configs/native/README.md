# Native Configs

Owner: `configs/native/README.md`.
State: canonical documentation.


## Purpose

Native model-shape configs live here. They are intentionally small JSON files
so the C++ trainer can parse them without a third-party JSON dependency.

## Contents

- [native_debug_bf16.json](native_debug_bf16.json): tiny verification shape
  used by routine native dense checks.
- [native_accepted_dense_bf16.json](native_accepted_dense_bf16.json): small
  real packed-cache dense target for the `accepted_training` proof run.
- [native_dense_20m_bf16_3070.json](native_dense_20m_bf16_3070.json):
  explicit dense BF16 20M-size RTX 3070 run shape for seq1024 real-data runs.
- [native_dense_40m_bf16_3070.json](native_dense_40m_bf16_3070.json):
  accepted dense BF16 40M-size RTX 3070 browser-demo shape. It is the native
  pair for `configs/training/dense_40m_accepted_3070.json`.
- [native_transformer_debug_bf16.json](native_transformer_debug_bf16.json):
  tiny verification shape for explicit native transformer training. It uses
  learned positional embeddings and untied embeddings.
- [decoder_debug_bf16.json](decoder_debug_bf16.json): tiny verification shape
  for explicit native decoder plumbing.
- [decoder_40m_bf16_3070.json](decoder_40m_bf16_3070.json): tied-embedding
  RTX 3070 decoder acceptance target once attention, backward, optimizer
  coverage, and KV-cache decode gates land.
- [decoder_140m_bf16_5090.json](decoder_140m_bf16_5090.json): RTX
  5090/Blackwell decoder profile target.
- [native_20m_bf16_3070.json](native_20m_bf16_3070.json): RTX 3070
  transformer profile target for future CUDA-forward profiling. It is not
  accepted transformer CUDA training.
- [native_40m_bf16.json](native_40m_bf16.json): legacy scratch 40M target
  shape for manual smoke runs. It is not a default.
- [native_120m_bf16_5090.json](native_120m_bf16_5090.json): RTX
  5090/Blackwell transformer profile target. It is not an acceptance baseline.

## Rules

- Keep tensor dimensions explicit.
- Routine verification must use the debug config, not the 40M shape.
- Dense configs size the implemented embedding-plus-LM-head path. Transformer
  profile configs do not imply dense parameter counts.
- Decoder acceptance configs use tied token embeddings and LM head. Untied
  decoder shapes are diagnostics only.
- Profile configs must satisfy `heads * head_dim == hidden_size` and
  `heads % kv_heads == 0`, but they do not promote transformer training.
