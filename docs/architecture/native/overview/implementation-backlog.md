# Native Implementation Backlog

Owner: `docs/architecture/native/overview/implementation-backlog.md`.
State: implementation backlog.

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

## Dense Diagnostic Surface

The dense 40M native browser diagnostics provide truthful dense artifact
status, local top-k logits, checksum stability, benchmark provenance, and no
chat claim for dense artifacts.

## Active Decoder Target

The active implementation target remains the 40M RTX 3070 decoder lane, but
the current code stage is not accepted training. The next active step is
stateful full-forward parity and small backward primitives with CTest evidence;
promotion still requires full decoder backward, optimizer coverage, contiguous
BF16 KV-cache decode, native runtime chat, accepted report fields, and two-hour
RTX 3070 evidence. cuDNN SDPA and NCCL stay after single-GPU correctness and
profiling.

## Research-Informed Order

The latest report, `tmp/deep-research-report (61).md`, modified
`2026-05-13`, keeps this durable order active:

1. Keep dense diagnostics and report fields truthful.
2. Keep historical partial decoder fields fenced from accepted claims.
3. Keep the decoder forward substrate wired into the actual training path.
4. Keep block-tensor backward and FP32 AdamW state covered by CTests that prove
   non-embedding block weights change.
5. Promote reports only after all decoder trainable tensors have optimizer
   coverage and checkpoint/export/logits checks pass.
6. Disclose accepted contiguous BF16 KV-cache decode names only when the
   executed route path and sidecar agree.
7. Add decode metrics: time to first token, decode tokens per second, queue
   wait, cache bytes, cache blocks allocated/reused/evicted, and sampler time.
8. Add continuous batching only after single-request KV-cache correctness.
9. Add large profiles only after the 40M RTX 3070 lane is accepted.

See [decoder/README.md](../decoder/README.md) for the same-model chat path. See
[transformer-cuda-plan.md](../cuda/transformer-cuda-plan.md) for the retained
experimental transformer path and current unsupported decode contract.
