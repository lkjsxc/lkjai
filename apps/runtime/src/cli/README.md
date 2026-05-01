# Runtime CLI

## Purpose

CLI modules implement local verification gates used by Docker Compose.

## Contents

- [corpus.rs](corpus.rs): committed SFT action validation.
- [docs.rs](docs.rs): README topology and Markdown link validation.
- [mod.rs](mod.rs): CLI module exports.
- [quality.rs](quality.rs): line-limit and forbidden-runtime checks.
- [topology.rs](topology.rs): durable tracked directory README checks.

## Rules

- Gate output is JSON where practical.
- Checks must be deterministic and runnable inside the verify container.
