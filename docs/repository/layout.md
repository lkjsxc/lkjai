# Repository Layout

## Root Entries

- `README.md`
- `LICENSE`
- `docs/`
- `native/`
- `training/`
- `corpus/`
- `configs/`
- `ops/`
- `data/`
- `compose.yaml`

## Source Layout

- `native/`: C++/CUDA train, serve, runtime, artifact, tokenizer, checks, and
  kernel code.
- `training/`: legacy migration notes only; product code does not live here.
- `corpus/generated/`: committed validated generated corpus artifacts.
- `corpus/sources/`: reviewed JSON source entries expanded into rows.
- `ops/docker/`: Dockerfiles.
- `docs/`: canonical documentation.
- `data/`: local untracked runtime and training artifacts.

## Protected Canon

- The docs canon is protected project intent.
- Files outside docs may be replaced when needed to satisfy the canon.
