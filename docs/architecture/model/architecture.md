# Decoder Architecture

## Block Contract

- Each block uses pre-norm residual structure.
- Attention uses RoPE and grouped-query attention.
- Feed-forward layers use SwiGLU.
- Norm layers use RMSNorm.
- Output logits are projected through tied embeddings.

## Attention Contract

- The architecture supports alternating full and local attention layers.
- Local attention uses a sliding window.
- Full attention layers preserve global information flow.
- CUDA training should use FlashAttention-compatible kernels when available.

## Research Translation

- Qwen-style hybrid attention informs the local/full alternation.
- Gemma-style edge-focused sizing informs the 512 MiB serving target.
- Nemotron-style MTP and MoE are documented future extensions, not v1 defaults.
