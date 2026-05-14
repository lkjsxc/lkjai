# Kernel Plan

Owner: `docs/operations/performance/measurement/kernel-plan.md`.
State: canonical documentation.


## Escalation Order

1. cuBLASLt for dense projections.
2. cuDNN frontend SDPA for BF16 causal GQA accepted attention after parity.
3. Correctness-first custom CUDA causal GQA as fallback and parity oracle.
4. CUTLASS paths for exact-shape fused patterns only after measurement.
5. Custom CUDA for measured cache, decode, sampler, and fusion hotspots.
6. CUDA Graph replay for stable decode and train buckets.
7. NCCL only after single-GPU native acceptance passes.

## Library Rules

- GEMM replacement is out of scope unless a profiler proves a library path is
  unavailable or wrong for the shape.
- Prefer cuBLASLt for GEMMs and use cuDNN SDPA as the first accepted BF16 GQA
  attention owner.
- Keep native CPU/reference checks for correctness, not product execution.
- The custom CUDA causal GQA path remains diagnostic fallback/oracle evidence
  and must not be reported as accepted attention.

## Native CUDA Entry Points

Custom CUDA is expected for:

- RMSNorm plus residual,
- SwiGLU glue around linear projections,
- RoPE application,
- decode-time KV cache update.

Triton is not part of the product path.
