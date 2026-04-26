# Agent Prompt

## Goal

Keep the runtime prompt short, tracked, and aligned with the training action
schema.

## Files

- Canonical system prompt: `prompts/codex-40m-system.txt`
- Inactive YOLO supplement: `prompts/codex-40m-yolo.txt`
- Runtime loader: `apps/runtime/src/agent/prompt.rs`

## Contract

- The canonical prompt uses the existing XML action schema with a required
  `<tool>` child.
- It must not introduce a `<type>` action schema.
- Simple greetings, thanks, and ordinary questions should finish directly with
  `<tool>agent.finish</tool>`.
- Multi-step work may use `<tool>agent.think</tool>` for a visible plan.
- Any `kjxlkj` mutation must first use
  `<tool>agent.request_confirmation</tool>`.
- The model must not repeat the same failed non-terminal action without new
  information.

## YOLO

`codex-40m-yolo.txt` is documentation-only unless a future runtime switch
explicitly appends it after the user asks for YOLO mode. It does not remove
license, privacy, safety, confirmation, push, deploy, or review constraints.
