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
- Historical container baseline before the native rewrite: PyTorch
  `2.5.1+cu124`.
- Prior long SFT run recorded about `37695` input tokens/sec.
- Prior short matrix showed `torch.compile` post-warm as the strongest
  measured direction.

## Current Speed Smoke

- Date: 2026-04-30.
- Image: historical `lkjai-train:latest` from PyTorch `2.11.0+cu128`.
- Case: `artifacts/benchmarks/speed-smoke/synthetic_gpu`.
- Result: about `69910` median input tokens/sec on two profiled synthetic
  microsteps after auto-batch selected batch `8`.
- Treat this as a model-path smoke benchmark; real packed-cache training still
  requires a full `train-speed-v1` run.

## Optimization Order

1. Measure the current training path with a bounded Compose benchmark.
2. Use the native CUDA image for this project.
3. Prefer library attention paths before custom kernels.
4. Rebuild packed caches as `uint16` because the vocabulary fits in 13 bits.
5. Sweep batch size, checkpointing, AMP, compile mode, and attention backend.
6. Promote the fastest stable setting into the committed training config.
7. Run the full training pipeline in a fresh data directory.

## Required Defaults

- Packed cache format: `lkjai-packed-cache-v2`.
- Packed token dtype: `uint16`.
- Default real loader candidate: batch-oriented mapped cache loading.
- Default compile mode for non-quick CUDA runs: `reduce-overhead`, unless the
  benchmark matrix selects a faster stable option.
- BF16 remains preferred when CUDA reports support.
- Serving decode reuses preallocated KV cache storage; training speed remains
  the first-order objective.

## Non-Goals

- Do not optimize by lowering the active context length.
- Do not add pretrained weights.
- Do not accept an optimization that prevents Compose verification from
  running on CPU.
