# Documentation Canon

`docs/` is the source of truth for runtime behavior, training behavior, and
repository policy.

## System Goal

- Build a from-scratch small dense language model for RTX 3070 8GB research.
- Run a local multi-turn agent loop with tools, memory, summaries, and strict
  JSON actions.
- Train tokenizer, corpus, model checkpoints, and agent-style supervision from
  local project artifacts instead of pretrained base weights.
- Keep this repository optimized for LLM-to-LLM maintenance.

## Global Rules

1. Keep one canonical owner for each contract.
2. Keep each docs directory to exactly one `README.md` plus multiple children.
3. Keep each docs file at `<= 300` lines.
4. Keep each authored source file at `<= 200` lines.
5. Prefer stable headings: `Goal`, `Contract`, `Defaults`, `Verification`.
6. Prefer exact commands, paths, and payload shapes over prose.
7. Remove stale behavior rather than preserving compatibility shims.
8. Docs-first workflow is mandatory: update docs, then implementation.

## Top-Level Sections

- [vision/README.md](vision/README.md): intent and LLM-oriented authoring rules.
- [getting-started/README.md](getting-started/README.md): setup, run, verify.
- [product/README.md](product/README.md): chat, tools, and API behavior.
- [architecture/README.md](architecture/README.md): agent, runtime, model, training.
- [operations/README.md](operations/README.md): Compose, deployment, quality gates.
- [repository/README.md](repository/README.md): layout, workflow, and file rules.
- [research/README.md](research/README.md): external model/training references.
- [decisions/README.md](decisions/README.md): accepted and rejected choices.

## Recommended Reading Order

1. [vision/purpose.md](vision/purpose.md)
2. [repository/workflow.md](repository/workflow.md)
3. [architecture/training/pipeline.md](architecture/training/pipeline.md)
4. [architecture/model/config.md](architecture/model/config.md)
5. [architecture/runtime/inference.md](architecture/runtime/inference.md)
6. [operations/compose.md](operations/compose.md)
7. [operations/quality.md](operations/quality.md)
8. [product/api.md](product/api.md)
