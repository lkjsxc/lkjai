# Decoder Architecture

## Accepted Traits

- The model is dense decoder-only.
- Blocks use pre-norm residual structure.
- Attention uses RoPE.
- Grouped-query attention is preferred.
- Feed-forward layers use SwiGLU or a close gated MLP variant.
- Norm layers use RMSNorm.
- Tied embeddings are acceptable when the upstream model uses them.

## RTX 3070 Constraint

- The default model must serve interactively on 8GB VRAM when quantized.
- Native context is capped operationally even if the model advertises more.
- Agent memory uses retrieval and summaries.

## Non-Defaults

- MoE is rejected for v1.
- Phase-1 multimodality is rejected.
- Custom Rust kernels for Qwen are rejected for v1 production serving.
- From-scratch pretraining remains research only.
