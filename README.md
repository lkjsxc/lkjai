# lkjai

`lkjai` is a docs-first from-scratch multi-turn agent research system for RTX
3070 8GB: train a small dense decoder locally, serve it through a separate
Python/Torch inference runtime, and orchestrate data-directory tool use,
memory, summaries, and XML actions in Rust.

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
- `inference` runs the Python/Torch OpenAI-compatible scratch inference service.
- `train` prepares corpus/tokenizer/checkpoint artifacts from scratch.
- Competency acceptance is behavioral eval pass rate `>= 80%`.
- Runtime data is mounted at `./data` for models, checkpoints, memory, runs, and
  the tool workspace.
- The mainline pretraining corpus target is 500M active English tokens staged
  under ignored `data/public-corpus/` with committed source recipes.
- Scratch training has two explicit objectives: `causal_lm_full` for full
  next-token pretraining and `assistant_masked_sft` for XML-action SFT.
- `TRAIN_MAX_STEPS` means optimizer steps; summaries separately report
  microsteps, optimizer steps, input tokens, and loss-bearing tokens.
- Optional Kimi synthetic corpus generation is documented in
  [docs/operations/training/kimi-corpus/README.md](docs/operations/training/kimi-corpus/README.md).

## Rule

When implementation and docs diverge, update docs first, then realign code.
