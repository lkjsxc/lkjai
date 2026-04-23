# CUDA Contract

## Required Behavior

- Training uses CUDA when `torch.cuda.is_available()` is true.
- Training stack uses local PyTorch scratch-model code.
- Mixed precision is enabled by default on CUDA.
- Gradient accumulation is used to fit 8 GiB VRAM.
- Activation checkpointing is supported by config.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `torch.compile` may be enabled by environment flag.
- FlashAttention may be used when installed and compatible.
- DataLoader pinned memory is enabled for CUDA runs.

## Fallback

- CPU smoke runs may exist for verification.
- Full scratch training is expected to prefer CUDA.
