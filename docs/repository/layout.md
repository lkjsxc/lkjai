# Layout Contract

## Required Root Entries

- `README.md`
- `LICENSE`
- `.gitignore`
- `.env.example`
- `build.zig`
- `build.zig.zon`
- `docs/`
- `src/`
- `web/`
- `scripts/`
- `Dockerfile`
- `Dockerfile.verify`
- `docker-compose.yml`
- `docker-compose.verify.yml`
- `verify.sh`

## Source Groups

- `src/main.zig`: app entrypoint.
- `src/server/`: HTTP handling.
- `src/agent/`: orchestration and librarian behaviors.
- `src/model/`: tokenizer, architecture, training, lightweighting.
- `src/storage/`: storage interfaces and adapters.
- `src/tests/`: focused test modules.

