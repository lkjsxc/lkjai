# Principles Contract

## Build Principles

1. Zig is the implementation language for core runtime, model stack, and orchestration.
2. Documentation contracts lead implementation.
3. CPU-only performance is treated as a first-class optimization target.
4. External CPU math libraries are allowed when they produce measurable gains.
5. Every critical behavior must be testable and script-verifiable.

## Agent Principles

- Decompose work into parallel subtasks when safe.
- Keep deterministic merge behavior for parallel outputs.
- Keep failure modes explicit and machine-readable.
- Keep librarian operations auditable and reversible where possible.

## Data Principles

- Canonical persistence target is PostgreSQL.
- Training corpora must be permissive, redistributable, and license-documented.
- Contract terms are singular and consistent across docs and code.

