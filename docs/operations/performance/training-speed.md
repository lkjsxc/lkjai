# Training Speed Contract

## Goal

Maximize tokens/sec for the existing `scratch-40m` model shape on the RTX 3070
8GB target:

- Vocabulary: `8192`
- Context: `1024`
- Layers: `10`
- Hidden size: `576`
- Attention heads: `8`
- KV heads: `2`
- FFN size: `1536`

Do not claim speedups from shrinking the model unless the experiment is clearly
marked as a separate preset.

## Baseline Facts

- Host target: NVIDIA RTX 3070, SM 8.6, 8 GiB VRAM.
- Accepted CUDA training evidence is native C++/CUDA dense BF16 today.
- Decoder is the product target, but full decoder CUDA reports and two-hour
  evidence are still required before it replaces dense in accepted aggregates.
- Historical PyTorch speed records are background data, not product
  optimization knobs.
- Current dense reports expose logits, grad-logits, hidden-gradient,
  cuBLASLt-workspace, allocator, and timing fields so throughput changes can be
  tied to the active CUDA path.

## Current Speed Smoke

- Date: 2026-04-30.
- Image: historical `lkjai-train:latest` from PyTorch `2.11.0+cu128`.
- Case: `artifacts/benchmarks/speed-smoke/synthetic_gpu`.
- Result: about `69910` median input tokens/sec on two profiled synthetic
  microsteps after auto-batch selected batch `8`.
- Treat this as a model-path smoke benchmark; real packed-cache training still
  requires a full `train-speed-v1` run.

## Optimization Order

1. Measure the current dense path with a bounded Compose benchmark.
2. Tune cuBLASLt plan choice and workspace within the accepted dense path.
3. Use stream-ordered allocation when supported and report allocator behavior.
4. Reduce phase-local synchronization by deferring CUDA-event timing to slot
   waits.
5. Reuse converted FP32 operands across microsteps when AdamW has not refreshed
   BF16 shadows.
6. Promote only settings that preserve dense report acceptance, resume
   equivalence, artifact checksums, and logits parity.
7. Run bounded 40M compatibility after debug-shape learning-control passes.

## Required Defaults

- Packed cache format: `lkjai-packed-cache-v2`.
- Packed token dtype: `uint16`.
- Default real loader candidate: batch-oriented mapped cache loading.
- Default native launch mode: plain launches; CUDA Graph replay is roadmap
  after dense and transformer launch order are stable.
- BF16 remains preferred when CUDA reports support.
- Dense serving decode remains unsupported in the accepted dense path; training
  speed remains the first-order objective until decoder CUDA acceptance.

## Non-Goals

- Do not optimize by lowering the active context length.
- Do not add pretrained weights.
- Do not promote transformer CUDA, decode, CUDA Graph, NCCL, or FP16 fallback
  work from dense speed evidence.
