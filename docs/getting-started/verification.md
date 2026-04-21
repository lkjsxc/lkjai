# Verification

## Goal

Verification proves the repository shape, small training pipeline, model export,
Rust inference wiring, web health route, and YOLO tool plumbing.

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
- A tiny model can be exported under the configured size limit.

## Non-Goal

- Verification does not train on the full ~3B-token corpus.
