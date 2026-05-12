# Kernel Plan

## Escalation Order

1. cuBLASLt for dense projections.
2. Correctness-first custom CUDA causal GQA for the first accepted decoder.
3. cuDNN frontend SDPA for eligible attention shapes after parity and timing.
4. CUTLASS paths for exact-shape fused patterns only after measurement.
5. Custom CUDA for measured cache, decode, sampler, and fusion hotspots.
6. CUDA Graph replay for stable decode and train buckets.
7. NCCL only after single-GPU native acceptance passes.

## Library Rules

- GEMM replacement is out of scope unless a profiler proves a library path is
  unavailable or wrong for the shape.
- Prefer cuBLASLt for GEMMs and use the custom CUDA causal GQA path as the
  first accepted attention owner.
- Keep native CPU/reference checks for correctness, not product execution.
- cuDNN SDPA may replace the first accepted attention path only after parity
  and timing have been measured for the active GQA shape.

## Native CUDA Entry Points

Custom CUDA is expected for:

- RMSNorm plus residual,
- SwiGLU glue around linear projections,
- RoPE application,
- decode-time KV cache update.

Triton is not part of the product path.
