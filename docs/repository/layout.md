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
- `training/`: Python tokenizer, scratch training, eval, manifest, and tests.
- `training/corpus/`: committed validated generated corpus artifacts.
- `training/corpus_sources/`: tagged JSON source entries expanded into rows.
- `docs/`: canonical documentation.
- `data/`: local untracked runtime and training artifacts.

## Protected Canon

- The docs canon is protected project intent.
- Files outside docs may be replaced when needed to satisfy the canon.
