# Decision Log

## Accepted Defaults

- Runtime orchestrator: Rust with axum.
- Inference runtime: separate Python/Torch OpenAI-compatible service.
- Serving model family: local scratch dense decoder.
- Training scale: `scratch-60m`, targeting 50-80M parameters.
- Training method: local PyTorch from random initialization.
- Tokenizer: local byte-level BPE.
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

## Rationale

- From-scratch training is the research question, even when weaker than
  pretrained workflows.
- Serving trained checkpoints through Python/Torch is more valuable than
  preserving a placeholder Rust inference stub.
- Rust remains the web and agent runtime direction.
- SQLite keeps memory simple, inspectable, and local.
- Health probes prevent silent fallback to fake responses.
- Real training requires real data; synthetic trajectories bootstrap behavior.
- Verification remains deterministic through dedicated checks rather than
  product-runtime dummy model fallbacks.
