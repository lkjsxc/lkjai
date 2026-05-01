# Agent Source

## Purpose

Agent modules parse XML actions, build prompts, run tools, persist transcripts,
and enforce confirmation and profile rules.

## Contents

- [action.rs](action.rs): XML action parser and field container.
- [chat.rs](chat.rs): bounded multi-step chat loop.
- [chat_actions.rs](chat_actions.rs): terminal, thinking, and confirmation
  action handlers.
- [confirmation.rs](confirmation.rs): pending mutation storage contract.
- [confirmation_flow.rs](confirmation_flow.rs): post-confirmation replay.
- [memory.rs](memory.rs): durable SQLite memory store.
- [mod.rs](mod.rs): agent facade and action validation.
- [prompt.rs](prompt.rs): model prompt construction.
- [schema.rs](schema.rs): chat request, response, and event types.
- [tool_fields.rs](tool_fields.rs): required and optional field helpers.
- [tool_local.rs](tool_local.rs): local filesystem and memory tool executor.
- [tool_registry.rs](tool_registry.rs): profile-based tool allowlist.
- [tool_remote.rs](tool_remote.rs): `kjxlkj` HTTP tool executor.
- [tool_runner.rs](tool_runner.rs): event logging around tool execution.
- [tool_summary.rs](tool_summary.rs): compact tool-call summaries.
- [tools.rs](tools.rs): typed tool call enum and dispatch.
- [transcript.rs](transcript.rs): transcript storage.
- [workspace.rs](workspace.rs): workspace path confinement.

## Rules

- Mutations must never execute without stored confirmation.
- Default profile behavior must stay narrow enough for the 40M model.
