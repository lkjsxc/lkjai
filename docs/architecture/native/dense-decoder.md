# Dense BF16 Native Contract

## Goal

Keep the current dense CUDA milestone honest while preserving the target
decoder-only transformer roadmap.

## Current Implemented Shape

The implemented product path is intentionally minimal:

- Token embedding table.
- LM-head table.
- FP32 master weights/state with BF16 CUDA shadows and BF16 exported weights.
- Packed-cache causal-LM batches.
- CUDA forward, backward, AdamW, checkpoint, export, and logits check.
- Native server artifact loading with explicit unsupported chat decode.

There are no implemented attention blocks, MLP blocks, KV cache, or
autoregressive decode in the accepted product path yet.

## Target Shape

- Preset: `scratch-40m`.
- Vocabulary: `8192`.
- Context: `1024`.
- Layers: `10`.
- Hidden size: `576`.
- Attention heads: `8`.
- KV heads: `2`.
- Head dimension: `72`.
- FFN size: `1536`.
- Default precision: BF16 weights with FP32 accumulation.
- Activation: SwiGLU.
- Positional encoding: RoPE.

## Target Library Ownership

- Embeddings and LM head are native tensors in the current dense artifacts.
- QKV/O projections, MLP projections, and norms are target transformer tensors.
- QKV, output, and FFN projections are target transformer work and will use
  cuBLASLt first when that path becomes accepted CUDA training.
- Attention uses cuDNN SDPA when the frontend headers, runtime, dtype, compute
  capability, head dimension, and mask mode are eligible.
- A correctness-first causal GQA fallback is allowed only until cuDNN SDPA is
  wired for the active shape.
- The active `head_dim=72` is BF16 SDPA-eligible because it is a multiple of `8`.

## Current Training State

- Optimizer: AdamW.
- Optimizer state: FP32 master weights and FP32 moments.
- Checkpoints include dense model tensors, optimizer tensors, trainer counters,
  resume metadata, config, and checksums.
- Resume restores FP32 masters and Adam moments, then rebuilds BF16 CUDA
  shadows before forward/backward continues.
- Exported serving artifacts omit optimizer tensors unless explicitly requested.

## Acceptance Milestone

The current dense CUDA path is accepted only when all of these are true:

- A fixed synthetic batch trains forward and backward without NaN loss.
- Packed-cache training consumes `lkjai-packed-cache-v2`.
- `lkjai-native-logits-check` validates finite `[1,V]` logits from an exported
  BF16 artifact. Repeated checks with the same seed/config/dataset must keep the
  exported logits checksum stable.
- Native server loads the artifact. Until decode lands, chat returns explicit
  unsupported-decode JSON for dense and transformer artifacts.
- GPU-required Compose verify passes on the native dense CUDA smoke.
- Capability JSON reports device CC, BF16 support, cuBLASLt, cuDNN, and SDPA
  eligibility.

The transformer path remains experimental until device-resident projections,
attention, MLP, norms, backward, optimizer state, and decode pass the same
artifact and runtime checks.

## Non-Goals

- Do not write custom GEMM before measuring cuBLASLt.
- Do not add tensor parallelism before single-GPU acceptance.
- Do not support Python `model.pt` as a product serving artifact.
- Do not keep transition-table generation in the product path.
