# kjxlkj Integration

## Goal

Prepare lkjai to become the server-side assistant for kjxlkj without changing
kjxlkj runtime code in this phase.

## Target Behavior

- Users invoke lkjai from a future kjxlkj chat surface.
- lkjai can search notes, read resources, summarize clusters, propose edits,
  create notes, update notes, and organize information.
- lkjai preserves kjxlkj visibility defaults and avoids exposing private
  resource content in public summaries.
- lkjai training data uses kjxlkj docs and API contracts as supervision.

## Phase Boundary

- This phase adds lkjai docs, corpus generation, and eval prompts for kjxlkj.
- A later kjxlkj phase adds HTTP routes, UI chat, and authenticated server-side
  command execution.
