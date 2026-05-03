# Kernel Roadmap

## Escalation Order

1. cuBLASLt for dense projections.
2. cuDNN frontend SDPA for eligible attention shapes.
3. CUTLASS paths for exact-shape fused patterns only after measurement.
4. Custom CUDA for measured cache, decode, sampler, and fusion hotspots.
5. CUDA Graph replay for stable decode and train buckets.
6. NCCL only after single-GPU native acceptance passes.

## Library Rules

- GEMM replacement is out of scope unless a profiler proves a library path is
  unavailable or wrong for the shape.
- Prefer cuBLASLt, cuDNN, and CUTLASS for standard dense math.
- Keep native CPU/reference checks for correctness, not product execution.
- A custom attention kernel is not accepted until cuDNN SDPA parity and timing
  have been measured for the active GQA shape.

## Native CUDA Entry Points

Custom CUDA is expected for:

- RMSNorm plus residual,
- SwiGLU glue around linear projections,
- RoPE application,
- decode-time KV cache update.

Triton is not part of the product path.
