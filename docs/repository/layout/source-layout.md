# Source Layout

Owner: `docs/repository/layout/source-layout.md`.
State: canonical documentation.

## Native Product Surface

- `native/src/`: C++/CUDA product implementation.
- `native/tests/`: native CTest contract and parity tests.
- `native/cmake/`: source and target wiring for native binaries.
- `ops/docker/`: CUDA, corpus, and verify images used by Compose.

## Documentation Surface

- `docs/`: protected canon for project intent, architecture, operations,
  product behavior, research summaries, and repository rules.
- Every docs directory has exactly one `README.md` and at least two children.
- Parent tables of contents link immediate children.

## Data And Config Surface

- `configs/native/`: native model-shape contracts.
- `configs/training/`: training-run contracts.
- `configs/corpus/` and `configs/curriculum/`: data and curriculum settings.
- `corpus/sources/`: reviewed source definitions expanded into rows.
- `corpus/generated/`: committed generated corpus artifacts used by checks.
