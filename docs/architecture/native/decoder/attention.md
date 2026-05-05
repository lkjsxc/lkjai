# Decoder Attention

## Acceptance Target

Accepted decoder attention is BF16 causal grouped-query attention for the
configured `heads`, `kv_heads`, and `head_dim`.

## Required Behavior

- Q and K use RoPE before score computation.
- The mask is strictly causal.
- GQA maps each query head to `head % kv_heads`.
- Accumulation uses FP32 or a vendor-backed equivalent with documented parity.
- Outputs match the host reference within the validation tolerance.
- The first accepted backend may be
  `cuda_causal_gqa_bf16_reference`; cuDNN SDPA is a later performance backend,
  not a blocker for first single-GPU acceptance.

## Current Status

The host reference implements causal GQA with RoPE. A CUDA parity hook now
checks BF16 causal GQA attention on deterministic MHA/GQA-style shapes, but the
trainer report must stay `accepted_cuda_training=false` until the full decoder
forward and backward path uses it.
