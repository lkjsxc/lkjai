# Transformer CUDA Roadmap

## Current State

The accepted native CUDA trainer is dense BF16 only: token embedding plus
LM-head, FP32 master weights and optimizer state, BF16 CUDA shadows,
packed-cache v2 ingestion, checkpoint/export, inspect, logits probes, and
report validation.

Transformer mode is retained for reference plumbing and artifact compatibility.
It is not accepted CUDA training. Reports must keep
`accepted_cuda_training=false`, `implementation_status=experimental`,
`transformer_status=experimental`, `forward_backend=host_reference`,
`backward_backend=host_surrogate`, and
`optimizer_backend=host_adamw_fp32` until real device-resident kernels replace
that path.

## Required Acceptance Order

1. Device-resident Q/K/V/O and MLP projections through cuBLASLt with tiny-shape
   CPU parity tests.
2. RMSNorm, RoPE, SwiGLU, cross-entropy, casts, and optimizer helpers as CUDA
   kernels with tolerance checks.
3. cuDNN SDPA forward for supported BF16 causal/GQA shapes, with explicit
   fallback reporting for unsupported shapes.
4. Transformer backward for projections, attention, norms, MLP, embeddings, and
   loss, with finite-difference or CPU-reference checks.
5. AdamW updates from FP32 state with BF16 shadow refresh and exact resume
   equivalence.
6. Export and logits checks for transformer artifacts produced by the accepted
   CUDA path.
7. Autoregressive decode with a contiguous native-owned KV cache and sampler.
8. CUDA Graph buckets only after train/decode shapes and launch order are
   stable.
9. NCCL only after the single-GPU transformer path passes correctness,
   profiling, and memory gates.

## Non-Goals For The Dense Milestone

- Do not promote host/reference transformer reports.
- Do not claim chat competency from logits probes or artifact loading.
- Do not add CUDA Graph, NCCL, or decode switches until the underlying
  correctness gates exist.
- Do not use RTX 5090 throughput numbers to relax the RTX 3070 acceptance gate.

## Decode Contract

Until milestone 7 lands, native server `/v1/chat/completions` must return HTTP
`422` with an unsupported-decode error and no `choices` field for both dense and
transformer artifacts. Successful `choices[0].message.content` responses are a
future decode milestone, not a current serving capability.
