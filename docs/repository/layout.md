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

- `src/`: Rust web, inference, agent, docs, and quality commands.
- `training/`: Python corpus, tokenizer, model, trainer, export, and tests.
- `docs/`: canonical documentation.
- `data/`: local untracked runtime and training artifacts.

## Protected Canon

- The docs canon is protected project intent.
- Files outside docs may be replaced when needed to satisfy the canon.
