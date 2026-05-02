# Kernel Policy

## Library First

- GEMM and linear layers use cuBLASLt unless profiling proves a better local
  path.
- cuDNN SDPA is future work because the current native image does not ship
  cudnn-frontend headers.
- CUTLASS is allowed for custom epilogues and exact-shape experiments.
- NCCL is not part of the first single-GPU acceptance gate.

## Custom CUDA

Custom kernels are accepted for:

- RMSNorm and residual fusion,
- RoPE application,
- correctness-first causal grouped-query attention,
- embedding lookup and backward,
- SwiGLU forward/backward,
- CE loss and backward,
- BF16/FP32 casts and AdamW,
- logits filtering,
- argmax or multinomial sampling,
- stop-token and `</action>` detection.

## Runtime Rules

- Use preallocated buffers for steady-state decode.
- Use CUDA memory pools for repeated allocation patterns.
- Use CUDA Graph replay for stable decode and train buckets.
- Keep FP32 accumulators for softmax, reductions, and optimizer state updates.
- Benchmark before replacing vendor GEMM or attention primitives.
