# Native Rewrite Strategy

## Target State

- Product training and serving run through native C++/CUDA binaries.
- The model API remains a separate local HTTP service.
- The Rust web runtime stays the agent orchestrator and HTTP client.
- Python is not a product training or inference dependency.
- Existing `.pt` checkpoints are not protected.

## Rewrite Boundary

Keep:

- Rust web runtime and agent loop.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` contracts.
- Docker Compose profiles.
- `kjxlkj` HTTP integration assumptions.
- Docs-first workflow and line-limit gates.

Replace:

- Python inference server.
- Python generation loop.
- Python scratch model execution.
- Python training step orchestration.
- Python packed-cache product reader.
- PyTorch checkpoint format.

## First Optimization Target

Optimize the RTX 3070 8GB path first:

- compute capability `8.6`,
- BF16-capable CUDA for accepted dense training; FP16 fallback is roadmap work,
- sequence length `1024`,
- `scratch-40m` decoder-only transformer shape.

Current dense CUDA handles the embedding plus LM-head foundation. Vendor
libraries and custom CUDA own the later transformer projections, attention,
cache, decode, sampler, and fusion work after correctness gates exist.
