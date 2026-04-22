# Verification

## Goal

Verification proves the repository shape, agent loop, structured tools, memory
store, web health route, and tuning fixture pipeline.

## Required Command

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Required Checks

- Rust formatting passes.
- Rust tests pass.
- Python smoke checks pass.
- Documentation topology passes.
- Markdown links pass.
- File line limits pass.
- No Node runtime files or commands are required.
- The fake model client can drive a tool call and final answer.
- Memory write and search pass deterministic checks.
- Tuning fixtures validate without large downloads.

## Non-Goal

- Verification does not download production model weights.
- Verification does not run QLoRA training.
