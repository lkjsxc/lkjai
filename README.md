# lkjai

`lkjai` is a docs-first CUDA C++ agent research system for RTX 3070 8GB:
train dense BF16 scratch models locally, serve native artifacts through an
OpenAI-compatible endpoint, and run the agent API, tools, memory, summaries,
tokenizer, cache builder, benchmarks, and verification through native C++.

Treat [docs/README.md](docs/README.md) as the only active canon for behavior,
architecture, operations, and repository policy. Route by contract owner and
task instead of reading the tree linearly.

## Start Here

- Canon root: [docs/README.md](docs/README.md)
- Current state and claim limits: [docs/current-state.md](docs/current-state.md)
- Quickstart: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Verification: [docs/getting-started/verification.md](docs/getting-started/verification.md)
- Compose contract: [docs/operations/compose.md](docs/operations/compose.md)
- Dense demo contract: [docs/product/dense-demo.md](docs/product/dense-demo.md)
- Scratch training contract: [docs/operations/training/runbooks/long-run.md](docs/operations/training/runbooks/long-run.md)
- Competency gate: [docs/operations/training/gates/competency-gate.md](docs/operations/training/gates/competency-gate.md)
- Decoder acceptance: [docs/architecture/native/decoder/training.md](docs/architecture/native/decoder/training.md)
- Benchmark output: [docs/operations/performance/contracts/benchmark-output.md](docs/operations/performance/contracts/benchmark-output.md)

## Current Shape

- Compose profiles: `inference`, `web`, `train`, `corpus`, `verify`.
- `web` runs the native C++ agent API runtime.
- `inference` loads native artifacts. Dense and transformer exports return
  explicit unsupported chat decode. Decoder exports may return choices through
  the current host-reference recompute bridge, but that is partial usability
  only and not accepted CUDA KV-cache serving.
- Direct OpenAI-compatible chat runs with
  `docker compose --profile inference up --build -d` and serves
  `http://127.0.0.1:8081/v1/chat/completions`. Set `MODEL_NAME` to an existing
  decoder export such as `decoder-2h-40m-3070` for chat choices.
- `train` runs the two-hour dense 40M packed-cache training profile from
  scratch; smoke checks are explicit native trainer invocations.
- Dense BF16 CUDA training is the accepted substrate. The decoder CUDA slice is
  partial: embeddings and LM head train, block forward is forward-only, block
  weights are not trained, and full decoder backward is not implemented.
- The dense 40M browser diagnostics expose local next-token logits, top-k
  output, checksums, and benchmark provenance through the merged native server.
- The active implementation target is `decoder_2h_40m_3070` on RTX 3070
  with real block-weight updates and native KV-cache decode.
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
- `corpus` is isolated for public corpus acquisition and validation; product
  train, serve, and verify paths remain native.

## Rule

When implementation and docs diverge, update docs first, then realign code.
