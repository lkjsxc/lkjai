# Decision Log

## Accepted Defaults

- Runtime orchestrator: native C++ HTTP service.
- Target serving shape: one native process owns `/v1/*` model routes and
  `/api/*` agent-runtime routes; loopback HTTP between native product
  components is transitional only.
- Inference runtime: merged native C++/CUDA service for `/api/*` and `/v1/*`;
  the direct inference profile is diagnostics-only.
- Serving model family: local scratch decoder-only transformer.
- Training scale: `scratch-40m` by default for the current corpus;
  `scratch-60m` remains a later target.
- Training method: local native C++/CUDA from random initialization.
- Tokenizer: local byte-level BPE with canonical XML-like tags added as single
  tokens.
- Memory backend: SQLite plus FTS lexical retrieval.
- Agent loop limit: `AGENT_MAX_STEPS=6`.
- Active context default: `1024` tokens.
- Runtime default requires a real native model artifact.
- Policy-file model mode is removed from the default product path.

## New Decisions

- Model health probe uses `GET /v1/models` with 5-second timeout.
- `Fake` model mode is test-only; production `ModelClient` always uses HTTP.
- Default corpus generation uses approved docs-derived rows only.
- Scratch chat formatting is owned by this repository.
- Fixed eval checks tokenizer, checkpoint, dataset, and loss artifacts.
- Behavioral eval checks real generated responses and owns competency.
- Compose verify is GPU-required and includes native dense CUDA smoke checks.
- Agent corpus default is 6,000 rows until reviewed non-LLM data exists.
- DPO is the first preference optimization phase.
- Runtime tool access is bounded to `TOOL_WORKSPACE_DIR`.
- kjxlkj integration starts as lkjai docs, corpus, and eval coverage before
  kjxlkj runtime routes.
- GPT/LLM-authored corpus packs are quarantined from default training.
- Default corpus generation is docs-derived until reviewed non-LLM data exists.
- Docker training reads a committed JSON config by default, with environment
  variables used only as explicit overrides.
- Model-facing prompt XML uses no attributes so canonical tags can remain
  atomic tokens.
- Product Rust and Python runtime/tooling paths are removed in favor of native
  C++/CUDA binaries and CTest gates.
- Native model artifacts use `lkjai-native-artifact` flat binary weights.
- Native serving failures are surfaced as non-success responses, not valid
  fallback XML actions.
- Transition-table train and serve code is removed from the product path.
- Dense BF16 CUDA is the only accepted native CUDA training path today;
  transformer CUDA training and autoregressive decode remain roadmap work.
- RTX 3070 8GB is the hardware acceptance gate; RTX 5090/Blackwell is a
  higher-throughput benchmark profile only.

## Rationale

- From-scratch training is the research question, even when weaker than
  pretrained workflows.
- Serving, training, runtime, and verification through native C++ removes the
  old mixed Rust/Python/CUDA split.
- SQLite keeps memory simple, inspectable, and local.
- Health probes prevent silent fallback to fake responses.
- Real training requires real data; synthetic trajectories bootstrap behavior.
- Verification remains deterministic through dedicated checks rather than
  product-runtime dummy model fallbacks.
