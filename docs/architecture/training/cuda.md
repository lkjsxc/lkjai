# CUDA Contract

## Required Behavior

- Training uses CUDA when the native trainer detects a usable device.
- Training stack uses local C++/CUDA scratch-model code.
- Training containers target CUDA `12.8` and cuDNN `9`.
- Mixed precision is enabled by default on CUDA.
- `TRAIN_AMP=auto` chooses BF16 when supported and FP16 otherwise.
- `TRAIN_AMP=fp16` uses native loss scaling.
- Batch size 2 with gradient accumulation 4 is the default 40M path.
- `TRAIN_BATCH_POLICY=oom_fallback` lets the 40M path reduce microbatch size
  when an 8GB GPU cannot hold the configured shape.
- Activation checkpointing is wired through `TRAIN_ACTIVATION_CHECKPOINT`.
- Checkpoint wrappers use `TRAIN_CHECKPOINT_PRESERVE_RNG=false` unless a
  dropout-sensitive experiment opts back in.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `TRAIN_COMPILE` is removed from the product path.
- `TRAIN_ATTENTION_BACKEND=auto` prefers native PyTorch SDPA unless a benchmark
  selects a faster backend.
- `TRAIN_ATTENTION_BACKEND=flash2` requires building the train image with
  `INSTALL_FLASH_ATTN=1`.
- `TRAIN_ATTENTION_BACKEND=sdpa_flash` forces PyTorch flash SDPA.
- Native CUDA and vendor-library attention paths own the mandatory baseline.
- DataLoader pinned memory is enabled for CUDA runs.
- Model training uses fused QKV and fused SwiGLU projections in the scratch
  decoder.

## Fallback

- CPU smoke runs may exist for verification.
- Full scratch training is expected to prefer CUDA.
