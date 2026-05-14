# Native Architecture

Owner: `docs/architecture/native/README.md`.
State: canonical documentation.


Use this subtree for native C++/CUDA train, serve, artifact, and acceptance
contracts.

## Status Canon

| Mode | Status | Public Role |
|---|---|---|
| `dense` | CUDA foundation | Accepted BF16 training substrate and benchmark continuity path. |
| `decoder` | Product target | Same-model train/export/serve goal; current CUDA scaffolding is partial until real block backward, CUDA KV-cache decode, and acceptance evidence land. |
| `transformer` | Reference-only | Host/reference parity and migration checks; not a public product mode. |

## Read This Section When

- You need the product runtime boundary after Python removal.
- You need the native artifact format.
- You need CUDA kernel ownership rules.
- You need train, export, and serve acceptance gates.

## Child Index

- [overview/README.md](overview/README.md): strategy, capability, failure
  policy, and backlog.
- [contracts/README.md](contracts/README.md): artifact, runtime, training, and
  contract inventory ownership.
- [cuda/README.md](cuda/README.md): CUDA stack, tensor, kernel, and transformer
  diagnostic plans.
- [dense/README.md](dense/README.md): accepted dense substrate and dense
  runtime contracts.
- [decoder/README.md](decoder/README.md): decoder product target, artifact,
  training, decode, and benchmark contracts.
