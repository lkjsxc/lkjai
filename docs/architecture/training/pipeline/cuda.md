# CUDA Contract

## Required Behavior

- Training uses CUDA when the native trainer detects a usable device.
- Training stack uses local C++/CUDA scratch-model code.
- Training containers target CUDA `12.8` and cuDNN `9`.
- The current accepted trainer is dense BF16 CUDA. It is the foundation and
  benchmark substrate, not the final product mode.
- `decoder` is the product training target. It remains partial until reports
  prove `decoder_cuda_slice=full_decoder`, CUDA attention/backward, contiguous
  BF16 KV-cache decode, and the decoder benchmark gate.
- Dense training uses FP32 master weights and Adam state, BF16 CUDA shadow
  tensors for forward/backward, FP32 accumulation, and BF16 export.
- Accepted reports must say `accepted_cuda_training=true`,
  `implementation_status=accepted`, and `dense_cuda_path=true`.
- `transformer` is retained as reference-only plumbing and must say
  `accepted_cuda_training=false`.
- Batch size 2 with gradient accumulation 4 is the default 40M config.
- `grad_accum` is implemented as multiple dense CUDA microsteps before one
  AdamW optimizer step.
- No pretrained base model or 4-bit adapter loading is used by default.

## Optional Acceleration

- `TRAIN_COMPILE`, `TRAIN_AMP`, `TRAIN_ATTENTION_BACKEND`, activation
  checkpointing, auto-batch, and CUDA Graph switches are backlog knobs until
  the native trainer implements and reports them.
- cuBLASLt transformer projections, cuDNN SDPA, fused pointwise kernels, and
  CUDA Graphs are target optimization work after the dense CUDA foundation is
  stable. Dense LM-head GEMM may use cuBLASLt today.

## Fallback

- CPU smoke runs may exist for local diagnosis only.
- Full scratch training is expected to prefer CUDA.
- Compose verify requires CUDA and fails when the native dense CUDA smoke cannot run
  on the detected GPU.
