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
- The cuDNN wrapper owns eligibility checks, workspace sizing, shape-plan
  caching, forward, and backward entry points.
- Eligibility requires BF16, causal masking, `heads % kv_heads == 0`,
  `head_dim % 8 == 0`, and supported Ampere/Ada head-dimension limits.
- Accumulation uses FP32 or a vendor-backed equivalent with documented parity.
- Outputs match the host reference within the validation tolerance.
- Accepted reports use `attention_backend=cudnn_sdpa_bf16_gqa` only when the
  cuDNN path actually executes.
- `cuda_causal_gqa_bf16_reference` remains reportable only as diagnostic
  fallback and parity oracle evidence.

## Library Fit

The accepted 40M shape is compatible with the cuDNN SDPA constraints used by
the repo:

- cuDNN SDPA supports grouped-query attention.
- BF16 head dimensions are multiples of `8`.
- Ampere and Ada support decode and backward for head dimension `<= 128`.
- The 40M RTX 3070 shape has `head_dim=72`, `heads=8`, and `kv_heads=2`.

Reference: <https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html>

## Current Status

The custom CUDA reference forward runs BF16 causal GQA attention between RoPE
and the O projection, and CTest checks deterministic MHA/GQA shapes against the
host reference. Trainer reports still stay `accepted_cuda_training=false` until
forward and backward training execute cuDNN SDPA for accepted attention.
