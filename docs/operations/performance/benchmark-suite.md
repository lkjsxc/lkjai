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
