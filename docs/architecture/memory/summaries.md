# Rolling Summaries

## Goal

- Keep long conversations useful without sending full transcripts every turn.
- Preserve decisions, user preferences, pending tasks, and tool outcomes.

## Trigger

- Summarize when recent transcript text exceeds `SUMMARY_TRIGGER_CHARS`.
- Default trigger is `12000` characters.

## Content

- Current user goal.
- Important constraints.
- Completed actions.
- Open tasks.
- Durable facts worth writing to memory.

## Storage

- Summaries are keyed by `run_id`.
- The latest summary is loaded before each model action.
- Summary updates are transcript events when visible behavior changes.
