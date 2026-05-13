# Native Rewrite Strategy

Owner: `docs/architecture/native/overview/strategy.md`.
State: canonical native strategy.

## Target State

- Product training and serving run through native C++/CUDA binaries.
- One native server process is the target owner for `/v1/*` inference and
  `/api/*` agent-runtime routes.
- The native C++ runtime owns agent orchestration, tools, transcripts, memory,
  direct model-engine calls, and optional `kjxlkj` integration.
- Rust and Python are not product, verification, benchmark, or tooling
  dependencies.
- Existing `.pt` checkpoints are not protected.

## Rewrite Boundary

Keep:

- OpenAI-compatible `/v1/models` and `/v1/chat/completions` contracts.
- Native `/api/chat`, `/api/model`, `/api/runs/{id}`, and `/healthz`
  contracts.
- Docker Compose profiles.
- `kjxlkj` typed machine API integration.
- Docs-first workflow and line-limit gates.

Replace:

- Rust web runtime and Rust tool crates.
- Python benchmark, diagnostics, report, and test harnesses.
- Python or PyTorch scratch model execution.
- Rust packed-cache builder and reader utilities.
- Cookie/session `kjxlkj` resource access.
- Required loopback HTTP between native runtime and native inference code.

## First Optimization Target

Optimize the RTX 3070 8GB path first:

- compute capability `8.6`,
- BF16-capable CUDA for accepted dense training; FP16 fallback is backlog work,
- sequence length `1024`,
- `scratch-40m` decoder-only transformer shape.

Current dense CUDA handles the embedding plus LM-head foundation. Vendor
libraries and custom CUDA own the later transformer projections, attention,
cache, decode, sampler, and fusion work after correctness gates exist.
