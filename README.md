# lkjai

`lkjai` is a docs-first CUDA C++ agent research system for RTX 3070 8GB:
train dense BF16 scratch models locally, serve native artifacts through an
OpenAI-compatible endpoint, and run the agent API, tools, memory, summaries,
tokenizer, cache builder, benchmarks, and verification through native C++.

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
- `web` runs the native C++ agent API runtime.
- `inference` loads native artifacts and currently returns explicit unsupported
  chat decode for dense exports.
- `train` runs native dense CUDA smoke or packed-cache training from scratch.
- Competency acceptance is behavioral eval pass rate `>= 80%`.
- Runtime data is mounted at `./data` for models, checkpoints, memory, runs, and
  the tool workspace.
- The mainline release pack target is 500M public English pretraining tokens
  plus 60M first-party XML-action SFT tokens.
- Scratch training has two explicit objectives: `causal_lm_full` for full
  next-token pretraining and `assistant_masked_sft` for XML-action SFT.
- Canonical XML-like prompt and action tags are single tokenizer tokens.
- `TRAIN_MAX_STEPS` means optimizer steps; summaries separately report
  microsteps, optimizer steps, input tokens, and loss-bearing tokens.
- Optional synthetic corpus artifacts remain under `corpus/`; generation tools
  are not part of the active product train or serve path.

## Rule

When implementation and docs diverge, update docs first, then realign code.
