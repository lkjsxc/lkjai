# Model Research

## Qwen3 1.7B

- Default serving model family.
- Dense causal decoder with Qwen3 agent/tool-use suitability.
- Apache-2.0 license.
- Source: <https://huggingface.co/Qwen/Qwen3-1.7B>

## Qwen3 0.6B

- Default local tuning model family.
- Small enough for RTX 3070 QLoRA experimentation.
- Apache-2.0 license.
- Source: <https://huggingface.co/Qwen/Qwen3-0.6B>

## Architecture Traits

- Dense decoder-only architecture.
- RoPE position encoding.
- Grouped-query attention.
- RMSNorm and gated MLP family.

## Boundary

- Larger models may be served remotely, but v1 local defaults must fit RTX 3070.
