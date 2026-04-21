# CUDA Contract

## Required Behavior

- Training uses CUDA when `torch.cuda.is_available()` is true.
- Mixed precision is enabled by default on CUDA.
- Gradient accumulation is used to fit 8 GiB VRAM.
- Activation checkpointing is supported by config.
- Tokenized memmaps minimize CPU-side work during training.

## Optional Acceleration

- `torch.compile` may be enabled by environment flag.
- FlashAttention may be used when installed and compatible.
- DataLoader pinned memory is enabled for CUDA runs.

## Fallback

- CPU smoke runs may exist for verification.
- Full training is not expected to be practical on CPU.
