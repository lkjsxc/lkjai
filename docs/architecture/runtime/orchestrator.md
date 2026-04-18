# Orchestrator Contract

## Goal

Coordinate parallel agent subtasks and deterministic result merge for chat and record workflows.

## Runtime Model

- Incoming requests enter a bounded queue.
- Worker lanes process queued tasks in parallel.
- Default target: `16` in-flight requests.
- Backpressure returns explicit overload errors.

## Determinism Rules

- Merge order is stable by task id.
- Trace output always includes queue depth and executed step count.
- Errors are propagated and typed; they are not silently dropped.

## API Integration

- `/api/chat` routes through orchestrator.
- Record operations may execute directly or as orchestrated tasks depending on complexity.

