# lkjai

`lkjai` is a docs-first small-LLM project: train in Python/CUDA, serve in
Rust/Candle, and operate through Compose-first local workflows.

Treat [docs/README.md](docs/README.md) as the only active canon for behavior,
architecture, operations, and repository policy.

## Start Here

- Canon root: [docs/README.md](docs/README.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Verification: [docs/getting-started/verification.md](docs/getting-started/verification.md)
- Compose contract: [docs/operations/compose.md](docs/operations/compose.md)
- Long-run training contract: [docs/operations/training/long-run.md](docs/operations/training/long-run.md)
- Competency gate: [docs/operations/training/competency-gate.md](docs/operations/training/competency-gate.md)

## Current Shape

- Compose profiles: `train`, `web`, `verify`.
- `train` profile default is a long-run preset targeting ~6 hours.
- Competency acceptance is fixed-eval pass rate `>= 80%`.
- Runtime data is mounted at `./data` for corpus/tokenizer/checkpoint/model/run artifacts.

## Rule

When implementation and docs diverge, update docs first, then realign code.
