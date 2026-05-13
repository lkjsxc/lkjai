# Memory Retrieval

Owner: `docs/architecture/memory/retrieval.md`.
State: target architecture.

## Goal

- Retrieve compact relevant context before each model action.
- Avoid using huge native context as the memory mechanism.

## Current Status

- The active runtime persists transcripts but does not implement memory search.
- `memory.search` remains a tool contract target until a durable store exists.

## Target Retrieval

- Use SQLite FTS lexical search.
- Query with the latest user message and compact run summary.
- Return at most `MEMORY_TOP_K=5` records by default.
- Include memory text in a dedicated prompt section.

## Ranking

- Prefer exact lexical matches.
- Prefer recent updates when lexical score is similar.
- Prefer run-scoped memory for the active run.

## Future Hook

- Vector embeddings may be added after lexical memory passes tests.
- Vector retrieval must not replace transcript persistence.
