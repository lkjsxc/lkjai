# Transformer Contract

## Goal

Use a modern dense transformer design that balances quality and CPU deployability.

## Canonical Profile

- Parameter budget: approximately `250M` parameters before quantization.
- Decoder-only autoregressive architecture.
- Rotary positional embeddings.
- RMSNorm.
- SwiGLU feed-forward layers.
- Grouped-query attention for lower inference cost.
- Weight tying between token embedding and output head.

## Extensibility

- Layer counts, heads, and hidden size are config-driven.
- Activation and normalization variants are enum-driven.
- Runtime exposes model metadata for artifact-size gate checks.

