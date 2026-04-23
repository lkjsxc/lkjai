# Purpose

## Goal

- Build a small local agentic AI system for RTX 3070 8GB.
- Prioritize multi-turn behavior, tool calling, memory handling, and an explicit
  plan-act-observe-revise loop.
- Train a small dense decoder-only language model from random initialization.
- Serve scratch-model artifacts through a separate local inference runtime.
- Use external memory and summaries instead of unrealistic native context.
- Improve behavior through tokenizer, corpus, language-model training, and
  agent-style supervised trajectories.
- Host the user-facing application with Rust and axum.

## Audience

- The primary operator is an LLM agent.
- Human maintainers inspect and steer the project through AI agents.
- Documentation must be easier to retrieve and edit than unstated code intent.

## Non-Goals

- Do not use a pretrained base model as the default path.
- Do not use QLoRA or LoRA adapters as the default path.
- Do not use Qwen, Gemma, Kimi, DeepSeek, or other recent pretrained models as
  runtime dependencies.
- Do not preserve compatibility with older code or docs.
- Do not add MoE in v1.
- Do not add phase-1 multimodality.
- Do not rely on huge native context windows for memory.
- Do not run Node for hosting, verification, or browser automation.
