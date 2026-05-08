# Dense BF16 Native Contract

## Goal

Keep the current dense CUDA target honest while preserving the target
decoder-only transformer backlog.

## Foundation, Reference, And Product

- Dense foundation: accepted BF16 CUDA training for token embeddings and LM head
  with FP32 AdamW state, packed-cache input, checkpoint/export, logits checks,
  and explicit unsupported chat decode for dense artifacts.
- Decoder reference backend:
  `cuda_causal_gqa_bf16_reference` is a correctness-first CUDA attention path
  that may satisfy first full-decoder acceptance; it is not the dense
  foundation and not the preferred performance endpoint.
- Product decoder target: `configs/native/decoder_40m_bf16_3070.json` with
  `configs/training/decoder_2h_40m_3070.json`.
- Host-reference recompute decode is partial serving evidence only; accepted
  serving requires native contiguous BF16 KV-cache decode.

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
- Canonical first acceptance config:
  `configs/native/decoder_40m_bf16_3070.json`.
- Canonical first acceptance training config:
  `configs/training/decoder_2h_40m_3070.json`.
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
- First full decoder acceptance may use the correctness-first
  `cuda_causal_gqa_bf16_reference` backend.
- cuDNN SDPA is the later performance backend when frontend integration, dtype,
  capability, head dimension, and mask-mode parity are complete.
- The active `head_dim=72` is BF16 SDPA-eligible because it is a multiple of `8`.
- CUTLASS, CUDA Graphs, TensorRT, TensorRT-LLM, and NCCL are profiling,
  serving, or later scale tracks after native single-GPU decoder acceptance.

## Current Training State

- Optimizer: AdamW.
- Optimizer state: FP32 master weights and FP32 moments.
- Checkpoints include dense model tensors, optimizer tensors, trainer counters,
  resume metadata, config, and checksums.
- Resume restores FP32 masters and Adam moments, then rebuilds BF16 CUDA
  shadows before forward/backward continues.
- Exported serving artifacts omit optimizer tensors unless explicitly requested.

## Acceptance Target

The current dense CUDA path is accepted only when all of these are true:

- A fixed synthetic batch trains forward and backward without NaN loss.
- Packed-cache training consumes `lkjai-packed-cache`.
- `lkjai-native-logits-check` validates finite `[1,V]` logits from an exported
  BF16 artifact. Repeated checks with the same seed/config/dataset must keep the
  exported logits checksum stable.
- Native server loads the artifact. Until decode lands, chat returns explicit
  unsupported-decode JSON for dense and transformer artifacts.
- GPU-required Compose verify passes on the native dense CUDA smoke.
- Capability JSON reports device CC, BF16 support, cuBLASLt, cuDNN, and SDPA
  eligibility.

The decoder product path remains unaccepted until device-resident projections,
attention, MLP, norms, backward, optimizer state, tied embedding aliasing, and
KV-cache decode pass the same artifact and runtime checks.

## Non-Goals

- Do not write custom GEMM before measuring cuBLASLt.
- Do not add tensor parallelism before single-GPU acceptance.
- Do not support Python `model.pt` as a product serving artifact.
- Do not keep transition-table generation in the product path.
