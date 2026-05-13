# Memory Store

Owner: `docs/architecture/memory/store.md`.
State: target architecture.

## Files

- Run transcripts stay as JSONL under `data/agent/runs/`.
- Runtime code creates missing transcript directories.
- Durable memory search and summary storage are target work, not implemented
  runtime state.

## Current Record

- Each event is one JSON object per line.
- `kind` records `user`, `assistant`, `error`, and future tool/memory events.
- `content` stores the visible event text.
- `timestamp` is UTC RFC 3339-like text.
- `step` and `tool` are present only when applicable.

## Future Memory Record

- `id`: stable row id.
- `scope`: `global` or `run`.
- `run_id`: nullable run id.
- `content`: memory text.
- `created_at`: RFC 3339 timestamp.
- `updated_at`: RFC 3339 timestamp.

## Write Policy

- The active runtime does not expose `memory.write`.
- When memory write support is added, the agent may write memory only through
  `memory.write`.
- Memory writes must be logged as `memory_write` events.
- Human-visible transcripts must show when memory changes.
