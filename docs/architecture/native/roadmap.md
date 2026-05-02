# Native Implementation Roadmap

## Goal

Replace the current CPU-reference transformer slice with a device-resident
BF16 training and serving engine without changing the external Rust/runtime
contracts.

## Milestones

| Order | Milestone | Required output |
|---:|---|---|
| 1 | Device substrate | dtype-aware tensors, stream/handle context, memory accounting, copy tests |
| 2 | Capability probe | JSON reports CC, BF16, cuBLASLt, cuDNN, SDPA, and async allocation eligibility |
| 3 | Typed artifacts | explicit tensor metadata, config checksum, optimizer checkpoint state |
| 4 | Dense forward | cuBLASLt-backed projections with reference-logit parity on tiny models |
| 5 | Fused kernels | RMSNorm, RoPE, SwiGLU, CE loss, and cast kernels with tolerance tests |
| 6 | Attention | cuDNN SDPA train/prefill path plus GQA/mask parity tests |
| 7 | Real backward | projection, attention, norm, MLP, loss, and embedding gradients |
| 8 | Optimizer | AdamW with FP32 state and BF16 refresh; FP16 scaler fallback |
| 9 | Resume/export | atomic snapshots, exact restart metadata, stripped serving export |
| 10 | Decode | autoregressive transformer decode, contiguous KV cache, sampler |
| 11 | Graphing | CUDA Graph buckets after shapes and launch order are stable |
| 12 | NCCL | single-node data parallel after single-GPU correctness and profiling |

## Acceptance Style

Each milestone must add a small executable or CTest path. A milestone is not
accepted by comments, TODOs, or benchmark scripts alone.

## Current Slice

The repository currently has:

- a CPU reference-style transformer forward path,
- a surrogate gradient trainer that proves artifact wiring,
- packed-cache v2 ingestion,
- AdamW-style parameter mutation for smoke verification,
- native artifact export and logits inspection,
- a CUDA BF16/library capability smoke,
- a dtype-aware CUDA tensor substrate with BF16 round-trip coverage.

That slice is useful for contracts, but it is not the final trainer.

## Next Code Target

The next implementation target is typed artifacts plus reference-vs-native
forward parity entrypoints for tiny debug configs. cuBLASLt replacement,
cuDNN SDPA, true backward, and decode stay behind those contracts.
