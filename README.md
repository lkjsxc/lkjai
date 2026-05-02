# lkjai

`lkjai` is a docs-first from-scratch multi-turn agent research system for RTX
3070 8GB: train a minimal dense BF16 CUDA model locally, load dense artifacts
through a native OpenAI-compatible runtime, and orchestrate
data-directory tool use, memory, summaries, and XML actions in Rust.

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
- Optional Kimi synthetic corpus tooling remains under `tools/` and is not part
  of the product train or serve path.

## Rule

When implementation and docs diverge, update docs first, then realign code.
