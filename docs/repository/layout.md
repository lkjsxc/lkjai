# Repository Layout

## Root Entries

- `README.md`
- `LICENSE`
- `docs/`
- `apps/`
- `native/`
- `training/`
- `corpus/`
- `configs/`
- `ops/`
- `tools/`
- `data/`
- `Cargo.toml`
- `Cargo.lock`
- `compose.yaml`

## Source Layout

- `apps/runtime/`: Rust web, model client, agent, memory, docs, and quality commands.
- `native/`: C++/CUDA train, serve, artifact, tokenizer, and kernel code.
- `training/`: legacy migration notes only; product Python does not live here.
- `corpus/generated/`: committed validated generated corpus artifacts.
- `corpus/sources/`: reviewed JSON source entries expanded into rows.
- `ops/docker/`: Dockerfiles.
- `tools/`: Kimi generation, benchmarks, diagnostics, reports, and experiments.
- `docs/`: canonical documentation.
- `data/`: local untracked runtime and training artifacts.

## Protected Canon

- The docs canon is protected project intent.
- Files outside docs may be replaced when needed to satisfy the canon.
