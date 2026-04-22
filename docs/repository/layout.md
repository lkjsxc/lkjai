# Repository Layout

## Root Entries

- `README.md`
- `LICENSE`
- `docs/`
- `src/`
- `training/`
- `data/`
- `Cargo.toml`
- `Cargo.lock`
- `Dockerfile`
- `Dockerfile.train`
- `Dockerfile.verify`
- `docker-compose.yml`
- `verify.sh`

## Source Layout

- `src/`: Rust web, model client, agent, memory, docs, and quality commands.
- `training/`: Python tuning dataset, eval, manifest, and tests.
- `docs/`: canonical documentation.
- `data/`: local untracked runtime and training artifacts.

## Protected Canon

- The docs canon is protected project intent.
- Files outside docs may be replaced when needed to satisfy the canon.
