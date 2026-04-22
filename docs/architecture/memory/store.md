# Memory Store

## Files

- Run transcripts stay as JSONL under `data/agent/runs/`.
- Agent memory lives under `data/agent/memory.sqlite3`.
- Runtime code creates missing directories and database tables.

## Tables

- `memories`: durable facts and preferences.
- `summaries`: rolling summaries keyed by run id.
- `memory_fts`: lexical search index over memory text.

## Memory Record

- `id`: stable row id.
- `scope`: `global` or `run`.
- `run_id`: nullable run id.
- `content`: memory text.
- `created_at`: RFC 3339 timestamp.
- `updated_at`: RFC 3339 timestamp.

## Write Policy

- The agent may write memory only through `memory.write`.
- Memory writes are logged as `memory_write` events.
- Human-visible transcripts must show when memory changes.
