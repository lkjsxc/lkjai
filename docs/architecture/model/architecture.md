# Scratch Decoder Architecture

## Accepted Traits

- The model is dense decoder-only.
- Blocks use pre-norm residual structure.
- Attention uses RoPE.
- Grouped-query attention is preferred.
- Feed-forward layers use SwiGLU or a close gated MLP variant.
- Norm layers use RMSNorm.
- Tied embeddings are the default.
- Weights are initialized locally; no pretrained tensors are loaded.

## RTX 3070 Constraint

- The default training preset targets 25-60M parameters.
- The default model must fit RTX 3070 8GB training experiments with gradient
  accumulation.
- Native context is capped operationally even if the model advertises more.
- Agent memory uses retrieval and summaries.

## Non-Defaults

- MoE is rejected for v1.
- Phase-1 multimodality is rejected.
- Pretrained serving models are rejected as defaults.
- QLoRA and LoRA adapters are rejected as defaults.
- Recent pretrained systems may inspire architecture choices only.
