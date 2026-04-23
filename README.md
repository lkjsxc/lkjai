# lkjai

`lkjai` is a docs-first from-scratch multi-turn agent research system for RTX
3070 8GB: train a small dense decoder locally, run a separate Rust inference
runtime, and orchestrate tool use, memory, summaries, and JSON actions in Rust.

Treat [docs/README.md](docs/README.md) as the only active canon for behavior,
architecture, operations, and repository policy.

## Start Here

- Canon root: [docs/README.md](docs/README.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Verification: [docs/getting-started/verification.md](docs/getting-started/verification.md)
- Compose contract: [docs/operations/compose.md](docs/operations/compose.md)
- Scratch training contract: [docs/operations/training/long-run.md](docs/operations/training/long-run.md)
- Competency gate: [docs/operations/training/competency-gate.md](docs/operations/training/competency-gate.md)

## Current Shape

- Compose profiles: `inference`, `web`, `train`, `verify`.
- `web` runs the Rust agent orchestrator.
- `inference` runs the Rust OpenAI-compatible scratch inference service.
- `train` prepares corpus/tokenizer/checkpoint artifacts from scratch.
- Competency acceptance is fixed agent eval pass rate `>= 80%`.
- Runtime data is mounted at `./data` for models, checkpoints, memory, and runs.

## Rule

When implementation and docs diverge, update docs first, then realign code.
