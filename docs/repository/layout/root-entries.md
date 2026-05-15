# Root Entries

Owner: `docs/repository/layout/root-entries.md`.
State: canonical documentation.

## Durable Root Files

- `.dockerignore`: Docker build-context exclusions.
- `.env.example`: local environment template without secrets.
- `.gitignore`: ignored local output and generated artifacts.
- `Dockerfile.web`: static web image.
- `LICENSE`: project license.
- `README.md`: repository entrypoint and quick operation map.
- `compose.yaml`: canonical Compose services and profiles.

## Durable Root Directories

- `artifacts/`: local artifact handoff notes and ignored artifact payloads.
- `configs/`: native and training configuration contracts.
- `corpus/`: committed corpus fixtures, sources, and generated rows.
- `data/`: local runtime, training, model, and verification output.
- `docs/`: canonical documentation and project intent.
- `native/`: C++/CUDA train, serve, tokenizer, runtime, checks, and kernels.
- `ops/`: Dockerfiles, host helpers, and corpus acquisition tooling.
- `scripts/`: agent and utility scripts.
- `training/`: migration notes and training orientation, not product code.
- `web/`: static browser diagnostics and sandbox frontend assets.
