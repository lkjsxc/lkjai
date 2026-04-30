# Kernel Roadmap

## Escalation Order

1. cuBLASLt for dense projections.
2. cuDNN graph or CUTLASS paths for stable fused patterns.
3. Custom CUDA for measured cache, decode, sampler, and fusion hotspots.
4. CUDA Graph replay for stable decode and train buckets.
5. NCCL only after single-GPU native acceptance passes.

## Library Rules

- GEMM replacement is out of scope unless a profiler proves a library path is
  unavailable or wrong for the shape.
- Prefer cuBLASLt, cuDNN, and CUTLASS for standard dense math.
- Keep native CPU/reference checks for correctness, not product execution.

## Triton Entry Points

Custom CUDA is expected for:

- RMSNorm plus residual,
- SwiGLU glue around linear projections,
- RoPE application,
- decode-time KV cache update.

Triton is not part of the product path.
