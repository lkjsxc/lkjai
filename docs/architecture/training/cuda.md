# CUDA Contract

## Required Behavior

- Training uses CUDA when `torch.cuda.is_available()` is true.
- Training stack uses local PyTorch scratch-model code.
- Training containers target PyTorch `2.11.0` with CUDA `12.8`.
- Mixed precision is enabled by default on CUDA.
- `TRAIN_AMP=auto` chooses BF16 when supported and FP16 otherwise.
- `TRAIN_AMP=fp16` uses a GradScaler.
- Batch size 2 with gradient accumulation 4 is the default 40M path.
- `TRAIN_BATCH_POLICY=oom_fallback` lets the 40M path reduce microbatch size
  when an 8GB GPU cannot hold the configured shape.
- Activation checkpointing is wired through `TRAIN_ACTIVATION_CHECKPOINT`.
- Checkpoint wrappers use `TRAIN_CHECKPOINT_PRESERVE_RNG=false` unless a
  dropout-sensitive experiment opts back in.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `TRAIN_COMPILE=auto` enables `torch.compile(..., mode="reduce-overhead")`
  on CUDA non-quick runs after warmup.
- `TRAIN_ATTENTION_BACKEND=auto` prefers native PyTorch SDPA unless a benchmark
  selects a faster backend.
- `TRAIN_ATTENTION_BACKEND=flash2` requires FlashAttention-2.
- `TRAIN_ATTENTION_BACKEND=sdpa_flash` forces PyTorch flash SDPA.
- Native PyTorch scaled-dot-product attention remains the mandatory baseline.
- DataLoader pinned memory is enabled for CUDA runs.

## Fallback

- CPU smoke runs may exist for verification.
- Full scratch training is expected to prefer CUDA.
