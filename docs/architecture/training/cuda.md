# CUDA Contract

## Required Behavior

- Training uses CUDA when `torch.cuda.is_available()` is true.
- Mixed precision is enabled by default on CUDA.
- Gradient accumulation is used to fit 8 GiB VRAM.
- Activation checkpointing is supported by config.
- QLoRA 4-bit loading is the default agent tuning path.

## Optional Acceleration

- `torch.compile` may be enabled by environment flag.
- FlashAttention may be used when installed and compatible.
- DataLoader pinned memory is enabled for CUDA runs.

## Fallback

- CPU smoke runs may exist for verification.
- Full QLoRA tuning is not expected to be practical on CPU.
