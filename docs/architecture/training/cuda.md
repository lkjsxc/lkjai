# CUDA Contract

## Required Behavior

- Training uses CUDA when `torch.cuda.is_available()` is true.
- Training stack uses local PyTorch scratch-model code.
- Mixed precision is enabled by default on CUDA.
- `TRAIN_AMP=auto` chooses BF16 when supported and FP16 otherwise.
- `TRAIN_AMP=fp16` uses a GradScaler.
- Gradient accumulation is used to fit 8 GiB VRAM.
- Activation checkpointing is wired through `TRAIN_GRADIENT_CHECKPOINTING`.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `torch.compile` may be enabled by environment flag.
- Native PyTorch scaled-dot-product attention is used. Native GQA is preferred
  when available, with a repeat-K/V fallback.
- DataLoader pinned memory is enabled for CUDA runs.

## Fallback

- CPU smoke runs may exist for verification.
- Full scratch training is expected to prefer CUDA.
