# Native Implementation Backlog

## Goal

Grow the current dense BF16 CUDA foundation into a device-resident decoder
training, serving, and agent runtime system without Rust or Python product
code.

## Acceptance Targets

| Order | Target | Required output |
|---:|---|---|
| 1 | Device substrate | dtype-aware tensors, stream/handle context, memory accounting, copy tests |
| 2 | Capability probe | JSON reports CC, BF16, cuBLASLt, cuDNN, SDPA, and async allocation eligibility |
| 3 | Typed artifacts | explicit tensor metadata, config checksum, optimizer checkpoint state |
| 4 | Dense foundation | BF16 embedding plus LM-head train/export/logits smoke |
| 5 | Fused kernels | RMSNorm, RoPE, SwiGLU, CE loss, and cast kernels with tolerance tests |
| 6 | Attention | correctness-first CUDA causal GQA plus GQA/mask parity tests |
| 7 | Real backward | projection, attention, norm, MLP, loss, and embedding gradients |
| 8 | Optimizer | AdamW with FP32 state and BF16 refresh; FP16 scaler fallback |
| 9 | Resume/export | atomic snapshots, exact restart metadata, stripped serving export |
| 10 | Native runtime | C++ `/api/chat`, tools, transcripts, memory, and model client |
| 11 | Decode | autoregressive decoder decode, contiguous KV cache, sampler |
| 12 | Graphing | CUDA Graph buckets after shapes and launch order are stable |
| 13 | NCCL | single-node data parallel after single-GPU correctness and profiling |

## Acceptance Style

Each target must add a small executable or CTest path. A target is not
accepted by comments, TODOs, or benchmark scripts alone.

## Current Slice

The repository currently has:

- dense BF16 CUDA embedding plus LM-head training,
- real gradient accumulation for the dense trainer,
- strict packed-cache metadata and bounds validation,
- packed-cache ingestion,
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
kernels, correctness-first CUDA causal GQA attention, decoder backward,
contiguous BF16 KV-cache decode, native runtime chat, accepted report fields,
and two-hour RTX 3070 evidence. cuDNN SDPA and NCCL stay after single-GPU
correctness and profiling.

## Research-Informed Order

The latest research report agrees with the repo canon: dense BF16 CUDA is the
foundation, but chat-capable product value depends on completing decoder
training and decode. Keep this order:

1. Wire the existing decoder forward substrate into the actual training path
   without changing acceptance fields.
2. Add block-tensor backward and FP32 AdamW state until at least one
   deterministic CTest proves a non-embedding block weight changes.
3. Promote the report only after all decoder trainable tensors have optimizer
   coverage and checkpoint/export/logits checks pass.
4. Replace host-reference recompute serving with contiguous BF16 KV-cache
   decode and disclose the accepted backend names in responses.
5. Add large profiles only after the 40M RTX 3070 lane is accepted.

See [decoder/README.md](decoder/README.md) for the same-model chat path. See
[transformer-cuda-plan.md](transformer-cuda-plan.md) for the retained
experimental transformer path and current unsupported decode contract.
