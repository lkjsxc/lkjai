# `lkj-150m` Config

## Defaults

- Model family: decoder-only Transformer.
- Target scale: approximately 150M parameters.
- Tokenizer: byte-level BPE.
- Vocabulary size: 32,000.
- Hidden size: 768.
- Layers: 18.
- Attention heads: 12.
- KV heads: 4.
- MLP: SwiGLU.
- Norm: RMSNorm.
- Position encoding: RoPE.
- Embeddings: tied token embedding and output projection.

## Context

- Default context length is 1024 for the RTX 3070 8 GiB baseline.
- Longer context is a configuration change, not a separate architecture.
- Training may reduce microbatch size when context increases.

## Size Rule

- The final serving export includes model weights, tokenizer, and config.
- The final serving export must be `<= 512 MiB`.
- Optimizer states and intermediate checkpoints are excluded from this limit.
