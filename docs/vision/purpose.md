# Purpose

## Goal

- Build a project that covers the full path from from-scratch LLM development
  to a local AI agent that can chat, run commands, browse websites, and read or
  write files.
- Keep the final serving artifact below 512 MiB.
- Use NVIDIA CUDA aggressively during data preparation, training, and inference
  wherever the selected libraries support it.
- Host the user-facing application with Rust and axum.

## Audience

- The primary operator is an LLM agent.
- Human maintainers inspect and steer the project through AI agents.
- Documentation must be easier to retrieve and edit than unstated code intent.

## Non-Goals

- Do not preserve compatibility with the previous Zig scaffold.
- Do not run Node for hosting, verification, or browser automation.
- Do not treat the tiny verification corpus as the full training corpus.
- Do not claim frontier-model capability from a 150M model.
