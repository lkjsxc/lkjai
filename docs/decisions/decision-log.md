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

## New Decisions

- Model health probe uses `GET /v1/models` with 5-second timeout.
- `Fake` model mode is test-only; production `ModelClient` always uses HTTP.
- Synthetic corpus generator produces >= 100 trajectories for real training.
- Chat template formatting uses `tokenizer.apply_chat_template`.
- `quick` preset runs real training with reduced steps instead of a no-op marker.
- Fixed eval checks adapter weight files and training loss metrics.

## Rationale

- Qwen3 provides small dense models with agent and tool-use suitability.
- GGUF quantization makes local serving realistic on RTX 3070 8GB.
- QLoRA makes local post-training feasible.
- SQLite keeps memory simple, inspectable, and local.
- Health probes prevent silent fallback to fake responses.
- Real training requires real data; synthetic trajectories bootstrap behavior.
- Verification remains deterministic through dedicated smoke checks rather than
  product-runtime dummy model fallbacks.
