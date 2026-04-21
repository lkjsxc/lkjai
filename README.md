# lkjai

`lkjai` is a docs-first project for building a small language model from
scratch, exporting it below 512 MiB, and hosting a local YOLO AI agent in a
Rust and axum web application.

Treat [docs/README.md](docs/README.md) as the only active canon for product
behavior, model design, training operations, runtime shape, and repository
rules.

## Start Here

- Canonical documentation root: [docs/README.md](docs/README.md)
- Local quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Verification: [docs/getting-started/verification.md](docs/getting-started/verification.md)
- Runtime and training profiles: [docs/operations/compose.md](docs/operations/compose.md)

## Current Shape

- CUDA PyTorch training pipeline.
- Rust and axum web application.
- Candle-based Rust inference path.
- Local-only host-YOLO agent tools.
- Docker Compose profiles for `train`, `web`, and `verify`.
- Root `data/` mount for corpora, tokenizers, checkpoints, model exports, runs,
  and agent transcripts.

## Rule

If implementation and documentation diverge, update the documentation canon
first and then realign implementation.
