# Purpose

## Goal

- Build a small local agentic AI system for RTX 3070 8GB.
- Prioritize multi-turn behavior, tool calling, memory handling, and an explicit
  plan-act-observe-revise loop.
- Serve a small dense decoder model through a local model server.
- Use external memory and summaries instead of unrealistic native context.
- Improve behavior through instruction tuning and tool trajectory tuning.
- Host the user-facing application with Rust and axum.

## Audience

- The primary operator is an LLM agent.
- Human maintainers inspect and steer the project through AI agents.
- Documentation must be easier to retrieve and edit than unstated code intent.

## Non-Goals

- Do not center v1 on from-scratch pretraining.
- Do not preserve compatibility with older code or docs.
- Do not add MoE in v1.
- Do not add phase-1 multimodality.
- Do not rely on huge native context windows for memory.
- Do not run Node for hosting, verification, or browser automation.
