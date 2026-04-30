# Decision Log

## Accepted Defaults

- Runtime orchestrator: Rust with axum.
- Inference runtime: separate native C++/CUDA OpenAI-compatible service.
- Serving model family: local scratch dense decoder.
- Training scale: `scratch-40m` by default for the current corpus;
  `scratch-60m` remains a later target.
- Training method: local native C++/CUDA from random initialization.
- Tokenizer: local byte-level BPE with canonical XML-like tags added as single
  tokens.
- Memory backend: SQLite plus FTS lexical retrieval.
- Agent loop limit: `AGENT_MAX_STEPS=6`.
- Active context default: `1024` tokens.
- Runtime default requires a real model API endpoint.
- Policy-file model mode is removed from the default product path.

## New Decisions

- Model health probe uses `GET /v1/models` with 5-second timeout.
- `Fake` model mode is test-only; production `ModelClient` always uses HTTP.
- Default corpus generation uses approved docs-derived rows only.
- Scratch chat formatting is owned by this repository.
- Fixed eval checks tokenizer, checkpoint, dataset, and loss artifacts.
- Behavioral eval checks real generated responses and owns competency.
- Immediate Compose verify is docs/test focused; training smoke is optional.
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
- Product Python training and inference are removed in favor of native
  C++/CUDA binaries.
- Native model artifacts use `lkjai-native-artifact-v1` flat binary weights.

## Rationale

- From-scratch training is the research question, even when weaker than
  pretrained workflows.
- Serving and training through native CUDA removes Python from the critical
  product path while preserving the Rust HTTP integration boundary.
- Rust remains the web and agent runtime direction.
- SQLite keeps memory simple, inspectable, and local.
- Health probes prevent silent fallback to fake responses.
- Real training requires real data; synthetic trajectories bootstrap behavior.
- Verification remains deterministic through dedicated checks rather than
  product-runtime dummy model fallbacks.
