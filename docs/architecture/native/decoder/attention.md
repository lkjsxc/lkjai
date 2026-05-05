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

## Current Status

The host reference implements causal GQA with RoPE. The CUDA forward substrate
does not yet include accepted attention, so reports must keep
`attention_backend=not_implemented` and `accepted_cuda_training=false`.
