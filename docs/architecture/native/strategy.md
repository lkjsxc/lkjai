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
- BF16 when supported, FP16 otherwise,
- sequence length `1024`,
- `scratch-40m` dense decoder shape.

Vendor libraries handle standard dense math. Custom CUDA is reserved for cache,
decode, sampler, and fusion work that the libraries do not cover cleanly.
