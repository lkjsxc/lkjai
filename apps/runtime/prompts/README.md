# Runtime Prompts

## Purpose

Prompt files define the model-facing system contract loaded by the Rust agent.

## Contents

- [codex-40m-system.txt](codex-40m-system.txt): active compact system prompt.
- [codex-40m-yolo.txt](codex-40m-yolo.txt): inactive supplement for future
  explicit YOLO-mode experiments.

## Rules

- Keep prompts aligned with `docs/architecture/agent/`.
- Do not list tools that the active runtime profile rejects.
- Keep the XML action schema stable unless docs and validators change first.
