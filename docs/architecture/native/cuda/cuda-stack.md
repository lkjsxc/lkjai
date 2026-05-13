# CUDA Stack

Owner: `docs/architecture/native/cuda/cuda-stack.md`.
State: canonical CUDA stack contract.

## Baseline

The initial native stack is deliberately narrow:

- OS: Ubuntu `24.04`.
- Compiler: GCC `13` or Clang `18` for C++20 host code.
- Build: CMake `3.27+` and Ninja.
- CUDA Toolkit: `12.8.1`.
- cuDNN: `9.x`, with `9.18.x` preferred until newer releases are profiled.
- GEMM: cuBLASLt from the pinned CUDA toolkit.
- Attention: cuDNN frontend SDPA is target transformer work after the dense
  CUDA foundation.
- Shape kernels: CUTLASS only after cuBLASLt/cuDNN measurements justify it.
- Distributed: NCCL only after single-GPU acceptance passes.

## CUDA Architecture Policy

Default native builds compile a consumer multi-arch set for Ampere, Ada, Hopper,
and Blackwell: `86-real;86-virtual;89-real;89-virtual;90-real;90-virtual;120-real;120-virtual`.

Selection precedence is:

1. Explicit `-DCMAKE_CUDA_ARCHITECTURES=...`.
2. `LKJAI_CUDA_ARCHS=...`.
3. The repo default above.

Use these focused builds when measuring specific targets:

```bash
LKJAI_CUDA_ARCHS='86-real;86-virtual' cmake -S native -B /tmp/lkjai-native-86 -G Ninja
LKJAI_CUDA_ARCHS='120-real;120-virtual' cmake -S native -B /tmp/lkjai-native-120 -G Ninja
```

## Hardware Gate

- Native BF16 requires compute capability `8.0+`.
- RTX 3070 is compute capability `8.6` and is the first optimization target.
- RTX 3070 8GB is the acceptance gate; RTX 5090/Blackwell is a benchmark
  profile only. See
  [hardware-profiles.md](../../../operations/performance/profiles/hardware-profiles.md).
- Older devices may use CPU diagnostics only; FP16 loss scaling is backlog work.
- CPU fallback must be visible in health/model JSON and is not a performance
  acceptance path.

## Precision Policy

- Forward weights and activations default to BF16.
- Reductions, softmax stats, loss stats, optimizer math, and checksums use FP32.
- Training keeps FP32 master weights, FP32 gradients, and FP32 AdamW moments.
- BF16 does not use a GradScaler.
- FP16 fallback and GradScaler support are not implemented in the accepted dense
  trainer.

## Vendor Library Ownership

- Current dense CUDA owns token embedding, LM-head GEMM, CE loss, AdamW,
  checkpoint, export, and logits check.
- Target transformer work assigns QKV/output/FFN projections to cuBLASLt and
  eligible attention to cuDNN SDPA before custom kernels replace measured
  hotspots.
- CUDA Graph replay is target work for stable static buckets.

## Upgrade Rule

Upgrade one stack layer at a time. A stack upgrade must include:

1. Native CTest pass.
2. Capability JSON before and after.
3. Training tokens/sec comparison on the active debug shape.
4. Decode latency comparison once transformer decode exists.
5. A rollback note in the training report if performance regresses.

## Official References

- NVIDIA CUDA Compiler Driver, CUDA 12.8 architecture names:
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/>
- NVIDIA Blackwell Tuning Guide, CUDA 12.8:
  <https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/>
- NVIDIA stream-ordered memory allocator:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html>
