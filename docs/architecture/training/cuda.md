# CUDA Contract

## Required Behavior

- Training uses CUDA when the native trainer detects a usable device.
- Training stack uses local C++/CUDA scratch-model code.
- Training containers target CUDA `12.8` and cuDNN `9`.
- The current accepted trainer requires BF16-capable CUDA and reports failure
  clearly when that capability is unavailable.
- Dense training uses FP32 master weights and Adam state, BF16 CUDA shadow
  tensors for forward/backward, FP32 accumulation, and BF16 export.
- Reports must say `dense_cuda_path=true`; otherwise the run is a failure for
  the accepted native path.
- Batch size 2 with gradient accumulation 4 is the default 40M config.
- `grad_accum` is implemented as multiple dense CUDA microsteps before one
  AdamW optimizer step.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `TRAIN_COMPILE`, `TRAIN_AMP`, `TRAIN_ATTENTION_BACKEND`, activation
  checkpointing, auto-batch, and CUDA Graph switches are roadmap knobs until
  the native trainer implements and reports them.
- cuBLASLt projections, cuDNN SDPA, fused pointwise kernels, and CUDA Graphs are
  target optimization work after the dense CUDA foundation is stable.

## Fallback

- CPU smoke runs may exist for local diagnosis only.
- Full scratch training is expected to prefer CUDA.
- Compose verify requires CUDA and fails when the native dense CUDA smoke cannot run
  on the detected GPU.
