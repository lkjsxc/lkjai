# Native Runtime

Use this subtree for the native C++/CUDA train and serve contracts.

## Status Canon

| Mode | Status | Public Role |
|---|---|---|
| `dense` | CUDA foundation | Accepted BF16 training substrate and benchmark continuity path. |
| `decoder` | Product target | Same-model train/export/serve goal; current CUDA slice is partial until full block kernels and acceptance evidence land. |
| `transformer` | Reference-only | Host/reference parity and migration checks; not a public product mode. |

## Read This Section When

- You need the product runtime boundary after Python removal.
- You need the native artifact format.
- You need CUDA kernel ownership rules.
- You need train, export, and serve acceptance gates.

## Child Index

- [strategy.md](strategy.md): rewrite boundary and migration order
- [artifact.md](artifact.md): native checkpoint and weight files
- [capability.md](capability.md): reusable CUDA capability JSON shape
- [contract-inventory.md](contract-inventory.md): stable native contracts,
  additive fields, diagnostics, and future-versioned surfaces
- [cuda-stack.md](cuda-stack.md): pinned CUDA, vendor library, and precision stack
- [dense-bf16-optimization.md](dense-bf16-optimization.md): accepted dense
  BF16 speedup contract and non-goals
- [dense-p0-runtime.md](dense-p0-runtime.md): accepted dense P0 runtime,
  report fields, and benchmark gate
- [dense-substrate.md](dense-substrate.md): accepted dense CUDA tuning surface,
  runtime tunables, and acceptance boundaries
- [dense-decoder.md](dense-decoder.md): dense foundation relationship to the
  decoder target
- [decoder/README.md](decoder/README.md): chat-capable decoder model kind,
  artifact, training, decode, and benchmark contracts
- [device-tensor.md](device-tensor.md): device tensor ownership and copy rules
- [failure-policy.md](failure-policy.md): unsupported, degraded, and failure behavior
- [roadmap.md](roadmap.md): staged native implementation backlog
- [runtime.md](runtime.md): HTTP server and inference behavior
- [transformer-cuda-roadmap.md](transformer-cuda-roadmap.md): current
  transformer limits and native CUDA acceptance order
- [training.md](training.md): native trainer ownership and data flow
- [kernels.md](kernels.md): CUDA library and custom-kernel rules
