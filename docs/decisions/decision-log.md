# Decision Log

## Accepted Defaults

- Runtime orchestrator: Rust with axum.
- Model server: llama.cpp OpenAI-compatible CUDA server.
- Serving model family: Qwen3 dense decoder.
- Serving scale: 1.7B quantized GGUF.
- Tuning scale: 0.6B QLoRA first.
- Memory backend: SQLite plus FTS lexical retrieval.
- Agent loop limit: `AGENT_MAX_STEPS=6`.
- Active context default: `4096` tokens.
- Runtime default requires a real model API endpoint.
- Policy-file model mode is removed from the default product path.

## Rationale

- Qwen3 provides small dense models with agent and tool-use suitability.
- GGUF quantization makes local serving realistic on RTX 3070 8GB.
- QLoRA makes local post-training feasible.
- SQLite keeps memory simple, inspectable, and local.
- Verification remains deterministic through dedicated smoke checks rather than
  product-runtime dummy model fallbacks.
