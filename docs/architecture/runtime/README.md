# Runtime Architecture

Use this subtree for the native C++ server that owns the agent API runtime,
inference path, and persistent runtime state.

## Read This Section When

- You need native API route ownership.
- You need direct model-engine or transitional model-client behavior.
- You need runtime data paths.

## Child Index

- [web.md](web.md): native runtime server contract
- [inference.md](inference.md): OpenAI-compatible model client behavior
- [storage.md](storage.md): data directory and transcript storage
- [workspace.md](workspace.md): data-directory tool workspace boundary
