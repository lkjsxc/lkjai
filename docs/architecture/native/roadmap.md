# Native Implementation Roadmap

## Goal

Grow the current dense BF16 CUDA foundation into a device-resident decoder
training, serving, and agent runtime system without Rust or Python product
code.

## Milestones

| Order | Milestone | Required output |
|---:|---|---|
| 1 | Device substrate | dtype-aware tensors, stream/handle context, memory accounting, copy tests |
| 2 | Capability probe | JSON reports CC, BF16, cuBLASLt, cuDNN, SDPA, and async allocation eligibility |
| 3 | Typed artifacts | explicit tensor metadata, config checksum, optimizer checkpoint state |
| 4 | Dense foundation | BF16 embedding plus LM-head train/export/logits smoke |
| 5 | Fused kernels | RMSNorm, RoPE, SwiGLU, CE loss, and cast kernels with tolerance tests |
| 6 | Attention | cuDNN SDPA train/prefill path plus GQA/mask parity tests |
| 7 | Real backward | projection, attention, norm, MLP, loss, and embedding gradients |
| 8 | Optimizer | AdamW with FP32 state and BF16 refresh; FP16 scaler fallback |
| 9 | Resume/export | atomic snapshots, exact restart metadata, stripped serving export |
| 10 | Native runtime | C++ `/api/chat`, tools, transcripts, memory, and model client |
| 11 | Decode | autoregressive decoder decode, contiguous KV cache, sampler |
| 12 | Graphing | CUDA Graph buckets after shapes and launch order are stable |
| 13 | NCCL | single-node data parallel after single-GPU correctness and profiling |

## Acceptance Style

Each milestone must add a small executable or CTest path. A milestone is not
accepted by comments, TODOs, or benchmark scripts alone.

## Current Slice

The repository currently has:

- dense BF16 CUDA embedding plus LM-head training,
- real gradient accumulation for the dense trainer,
- strict packed-cache v2 metadata and bounds validation,
- packed-cache v2 ingestion,
- AdamW parameter updates,
- native artifact export and logits inspection,
- a CUDA BF16/library capability smoke,
- a dtype-aware CUDA tensor substrate with BF16 round-trip coverage,
- decoder artifacts that copy and checksum the real byte-level BPE tokenizer,
- native decoder prompt serialization, tokenization, sampling, and stop checks,
- an experimental transformer reference path that reports
  `accepted_cuda_training=false`.

That slice is useful for contracts, but it is not the final trainer.

## Decoder Target

The next implementation target is accepted `decoder` CUDA training for the
40M RTX 3070 preset: device-resident cuBLASLt projections, fused pointwise
kernels, cuDNN SDPA, decoder backward, contiguous BF16 KV-cache decode, native
runtime chat, accepted report fields, and two-hour RTX 3070 evidence. NCCL
stays after single-GPU correctness and profiling.

See [decoder/README.md](decoder/README.md) for the same-model chat path. See
[transformer-cuda-roadmap.md](transformer-cuda-roadmap.md) for the retained
experimental transformer path and current unsupported decode contract.
