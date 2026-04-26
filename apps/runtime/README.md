# Runtime App

## Purpose

The runtime app is the Rust service that exposes chat, loads prompts, runs
agent tools, validates docs, and performs local quality checks.

## Contents

- [src/](src/): Rust library, CLI, web server, config, model client, and agent
  modules.
- [prompts/](prompts/): runtime prompt files used by the agent.
- [tests/](tests/): Rust integration tests.
- [Cargo.toml](Cargo.toml): crate manifest for package `lkjai`.

## Verification

Run through Compose from the repository root:

```bash
docker compose --profile verify up --build --abort-on-container-exit verify
```
