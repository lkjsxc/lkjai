# Runtime Web

## Purpose

The web module exposes the chat API, model-status API, and static browser UI.

## Contents

- [index.html](index.html): built-in static chat interface.
- [mod.rs](mod.rs): Axum routes and server wiring.

## Rules

- Keep the web layer thin.
- Agent behavior belongs in `../agent/`, not in route handlers.
