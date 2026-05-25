# Memory Retrieval

Owner: `docs/architecture/memory/retrieval.md`.
State: partial implementation contract.

## Goal

- Retrieve compact relevant context before each model action.
- Avoid using huge native context as the memory mechanism.

## Current Status

- The active runtime persists transcripts under `data/agent/runs/`.
- `memory.search` reads deterministic JSONL records under
  `data/agent/memory/`.
- Missing memory directories return an empty successful result.

## Target Retrieval

- The first implementation uses bounded lexical JSONL search.
- SQLite FTS is a later scaling target.
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
