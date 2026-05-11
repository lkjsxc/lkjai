# Benchmark Suite

## Purpose

Define the required measurement layers for native CUDA changes.

## Layers

| Layer | Scope | Required output |
|---|---|---|
| Substrate | GEMM, RMSNorm, RoPE, GQA, SwiGLU, CE loss, AdamW helpers, KV writes | Latency, bandwidth or tokens/sec, backend, shape, tolerance |
| Fixed train step | One fixed config, batch, sequence, and gradient accumulation | Step timing split and memory fields |
| Bounded training | Fixed token count or wall-clock cap | Loss trend, artifact checks, train report, exact command |
| Serving | Prefill and decode at batch 1, 4, and 8 | Prompt tokens/sec, ms/token, KV-cache backend, allocation policy |
| Scheduler | Mixed prompt and decode requests after KV correctness | Queue wait, active batch size, fairness, cache reuse, eviction count |

## Promotion Rule

A faster result is not accepted unless it preserves:

- finite loss or accepted numeric tolerance,
- unchanged or intentionally updated artifact contracts,
- exact command and config paths,
- git commit, GPU, driver, CUDA, cuDNN, backend fields, and workspace sizes,
- visible limitations when the path is partial or diagnostic.

## Backend Comparisons

- cuBLASLt remains the default GEMM owner unless a measured replacement wins on
  active shapes.
- cuDNN SDPA can replace the correctness-first attention path only after parity
  and timing evidence for the active GQA shape.
- CUTLASS and custom kernels need before/after evidence and retained tests.
- TensorRT-family engines are inference-only comparisons after native decode is
  accepted.

## Decoder Acceptance Measurements

Accepted decoder evidence must include these fields in the train or serving
report before any performance claim is treated as product evidence:

- `implementation_status=accepted`
- `accepted_cuda_training=true`
- `decoder_backward_backend` naming the CUDA block backward path
- `optimizer_backend` naming FP32 AdamW coverage for every trainable tensor
- `kv_cache_backend=cuda_contiguous_bf16`
- `decode_backend=cuda_kv_cache`
- `workspace_high_water_bytes`
- time to first token, decode tokens/sec, sampler time, queue wait,
  cache bytes, cache reuse, and cache eviction counters
- exact config path, packed-cache digest, artifact checksum, and command

Partial paths must keep explicit diagnostic names. A report that trains only
embeddings and the LM head is dense-substrate evidence, not accepted decoder
training evidence. Synthetic decoder-block deltas and host recompute decode
also remain diagnostic until real backward and CUDA KV-cache decode are proven.
