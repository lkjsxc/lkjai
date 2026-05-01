# Runtime Source

## Purpose

This directory contains the Rust runtime service, CLI gates, model client, and
agent orchestration code.

## Contents

- [agent/README.md](agent/README.md): XML-action loop, tools, memory, and
  transcript modules.
- [cli/README.md](cli/README.md): docs, corpus, and quality gate commands.
- [web/README.md](web/README.md): HTTP server and static chat UI.
- [config.rs](config.rs): environment-backed runtime configuration.
- [lib.rs](lib.rs): crate module exports.
- [main.rs](main.rs): CLI and web-service entrypoint.
- [model_client.rs](model_client.rs): OpenAI-compatible model HTTP client.

## Rules

- Keep source files at `<= 200` lines.
- Prefer small modules with contracts documented in `docs/`.
- Runtime behavior must follow the docs canon before tests are updated.
