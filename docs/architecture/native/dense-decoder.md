# BF16 Transformer Native Contract

## Goal

Use a real decoder-only transformer for native training and serving while keeping the
native artifact, runtime, and verification boundaries stable.

## Active Shape

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

## Library Ownership

- Embeddings, QKV/O projections, MLP projections, norms, and LM head are native
  tensors in `weights.lkjw`.
- QKV, output, and FFN projections use cuBLASLt first in the CUDA path.
- Attention uses a correctness-first causal GQA fallback in this slice.
- cuDNN SDPA is future work until cudnn-frontend headers are available.

## Training State

- Optimizer: AdamW.
- Optimizer state: FP32 master weights and FP32 moments.
- Checkpoints include model tensors, optimizer tensors, scheduler counters,
  resume metadata, and config.
- Exported serving artifacts omit optimizer tensors unless explicitly requested.

## Acceptance Milestone

The transformer path is accepted only when all of these are true:

- A fixed synthetic batch trains forward and backward without NaN loss.
- Packed-cache training consumes `lkjai-packed-cache-v2`.
- `lkjai-native-logits-check` validates finite `[1,V]` logits from an exported
  artifact.
- Native server loads the artifact and returns explicit unsupported-decode for
  chat until the autoregressive decode slice.
- GPU-required Compose verify passes on the native transformer smoke.

## Non-Goals

- Do not write custom GEMM before measuring cuBLASLt.
- Do not add tensor parallelism before single-GPU acceptance.
- Do not support Python `model.pt` as a product serving artifact.
- Do not keep transition-table generation in the product path.
