# Scratch Decoder Architecture

## Accepted Traits

- The current implemented model is a dense token embedding plus LM head.
- The target model is dense decoder-only.
- Target blocks use pre-norm residual structure.
- Target attention uses RoPE from precomputed per-layer tables.
- Target training uses fused QKV projection and fused SwiGLU gate/up projection.
- Grouped-query attention is preferred for the target transformer.
- Target feed-forward layers use SwiGLU or a close gated MLP variant.
- Target norm layers use RMSNorm.
- Dense continuity configs may tie embeddings; the first accepted decoder path
  uses tied token embeddings and LM head.
- Weights are initialized locally; no pretrained tensors are loaded.

## RTX 3070 Constraint

- The default training preset targets about 40M parameters while the committed
  corpus is around 26M tokenizer tokens.
- `scratch-60m` remains a later scale target, not the active default.
- The default model must fit RTX 3070 8GB training experiments with gradient
  accumulation.
- If the target shape does not fit, reduce sequence length before hidden size and
  document the accepted fallback in the training summary.
- Native context is capped operationally even if the model advertises more.
- Agent memory uses retrieval and summaries.

## Non-Defaults

- MoE is rejected for v1.
- Phase-1 multimodality is rejected.
- Pretrained serving models are rejected as defaults.
- QLoRA and LoRA adapters are rejected as defaults.
- Recent pretrained systems may inspire architecture choices only.
