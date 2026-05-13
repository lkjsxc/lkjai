# Decoder Attention

Owner: `docs/architecture/native/decoder/attention.md`.
State: acceptance target.

## Acceptance Target

Accepted decoder attention is cuDNN SDPA BF16 causal grouped-query attention
for the configured `heads`, `kv_heads`, and `head_dim`.

## Required Behavior

- Q and K use RoPE before score computation.
- The mask is strictly causal.
- GQA maps each query head with grouped division:
  `q_head / (heads / kv_heads)`.
- Accumulation uses FP32 or a vendor-backed equivalent with documented parity.
- Outputs match the host reference within the validation tolerance.
- Accepted reports use `attention_backend=cudnn_sdpa_bf16_gqa`.
- `cuda_causal_gqa_bf16_reference` remains reportable only as diagnostic
  fallback and parity oracle evidence.

## Current Status

The host reference implements causal GQA with RoPE. The custom CUDA forward
substrate runs BF16 causal GQA attention between RoPE and the O projection, and
CTest checks deterministic MHA and GQA shapes against the host reference.
Trainer reports still stay `accepted_cuda_training=false` until full decoder
forward and backward training uses cuDNN SDPA for accepted attention.
