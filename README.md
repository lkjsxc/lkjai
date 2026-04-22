# lkjai

`lkjai` is a docs-first local agentic AI system for RTX 3070 8GB: orchestrate
multi-turn tool use in Rust, serve a small dense model locally, and improve
behavior through post-training.

Treat [docs/README.md](docs/README.md) as the only active canon for behavior,
architecture, operations, and repository policy.

## Start Here

- Canon root: [docs/README.md](docs/README.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Verification: [docs/getting-started/verification.md](docs/getting-started/verification.md)
- Compose contract: [docs/operations/compose.md](docs/operations/compose.md)
- Agent tuning contract: [docs/operations/training/long-run.md](docs/operations/training/long-run.md)
- Competency gate: [docs/operations/training/competency-gate.md](docs/operations/training/competency-gate.md)

## Current Shape

- Compose profiles: `model`, `web`, `train`, `verify`.
- `web` runs the Rust agent orchestrator.
- `model` runs the local OpenAI-compatible model server.
- Competency acceptance is fixed agent eval pass rate `>= 80%`.
- Runtime data is mounted at `./data` for models, adapters, memory, and runs.

## Rule

When implementation and docs diverge, update docs first, then realign code.
