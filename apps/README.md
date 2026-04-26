# Apps

## Purpose

Application entry points live here. Each child directory owns one runnable
application or service boundary.

## Contents

- [runtime/](runtime/): Rust web server, agent loop, model client, prompts, and
  runtime tests.

## Rules

- Keep application code under child directories.
- Shared project intent belongs in `docs/`.
- Training code belongs in `training/package/`.
