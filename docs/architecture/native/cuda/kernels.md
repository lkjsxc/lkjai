# Kernel Policy

## Library First

- GEMM and linear layers use cuBLASLt unless profiling proves a better local
  path.
- The first accepted attention path may be the custom causal GQA BF16 reference
  kernel when parity passes.
- cuDNN frontend SDPA remains the preferred performance target when headers,
  runtime, dtype, and model shape are eligible.
- CUTLASS is allowed for custom epilogues and exact-shape experiments after
  cuBLASLt or cuDNN measurements justify it.
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
- Do not replace the reference attention path with a custom FlashAttention-style
  kernel before cuDNN SDPA parity and timing have been measured on the active
  GQA shape.
