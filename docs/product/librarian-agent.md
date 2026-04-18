# Librarian Agent Contract

## Goal

Support record-centric assistant behavior for cataloging, retrieval, summarization, and update planning.

## Core Operations

- `upsert_record`
- `delete_record`
- `search_records`
- `summarize_records`
- `plan_followups`

## Orchestration Rules

- Agent decomposes each request into subtasks.
- Subtasks execute in parallel worker lanes when independent.
- Merge stage ranks and unifies partial results deterministically.
- Default concurrency target is `16` in-flight requests with queueing.

## Response Rules

- Include short operator-facing answer.
- Include trace metadata for queue depth and parallel step count.
- Keep failure details explicit; no silent fallback behavior.

