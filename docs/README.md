# lkjai Documentation Canon

## Goal

`docs/` is the single source of truth for `lkjai` behavior, architecture, operations, and repository policy.

## Rules

- Update docs contracts before changing implementation.
- Keep each docs file at `<= 300` lines.
- Keep each source file at `<= 200` lines.
- Every docs directory must contain exactly one `README.md` TOC and multiple child files/directories.
- Prefer short declarative bullets over long narrative text.

## Top-Level Index

- [getting-started/README.md](getting-started/README.md): setup and verification entry path.
- [vision/README.md](vision/README.md): product goal, principles, and LLM-first rules.
- [product/README.md](product/README.md): private console behavior, API, and librarian-agent UX.
- [architecture/README.md](architecture/README.md): model, training, orchestration, and storage contracts.
- [operations/README.md](operations/README.md): compose runtime and verification gates.
- [repository/README.md](repository/README.md): layout, workflow, and line-limit policy.

## Locked Decisions

- Language: Zig.
- Training target: strict from-scratch (random initialization).
- Runtime target: CPU-only.
- CPU acceleration policy: external native math libraries are allowed.
- Model profile: dense transformer around `~250M` parameters before quantization.
- Deployment artifact policy: quantized inference artifact must be `<= 512 MiB`.
- Domain: English general-purpose with librarian/catalog focus.
- Access model: single-operator private console.
- Persistence target: PostgreSQL.
- Concurrency target: orchestration supports at least `16` concurrent requests with queueing.

