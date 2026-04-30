# Kernel Roadmap

## Escalation Order

1. Native PyTorch SDPA.
2. PyTorch SDPA with explicit flash backend.
3. FlashAttention-2 in a dedicated `INSTALL_FLASH_ATTN=1` benchmark image
   where install, GPU, dtype, and shape support are valid.
4. `torch.compile` with stable static shapes.
5. Triton kernels for measured elementwise or cache-update hotspots.
6. Hand-written CUDA only after profiler evidence.

## Library Rules

- GEMM replacement is out of scope unless a profiler proves a library path is
  unavailable or wrong for the shape.
- Prefer cuBLAS, cuDNN, SDPA, and FlashAttention for standard attention math.
- Keep native PyTorch SDPA as the correctness baseline.

## Triton Entry Points

Add Triton only after the benchmark matrix shows a remaining hotspot in one of:

- RMSNorm plus residual,
- SwiGLU glue around linear projections,
- RoPE application,
- decode-time KV cache update.

## CUDA Entry Points

Hand-written CUDA is reserved for:

- persistent batch-1 decode,
- bespoke cache movement,
- graph-captured fixed-shape serving paths,
- measured bottlenecks that PyTorch, FlashAttention, and Triton do not remove.
