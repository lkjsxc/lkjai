# Documentation Canon

`docs/` is the only active canon for `lkjai`.

## System Goal

- Train and serve a commercial-safe scratch model. The active default is about
  `40M` parameters for the current corpus; `60M` remains a later scale target.
- Keep the runtime LLM-readable: paired XML-like prompt sections and one
  XML-like assistant action with child tags only.
- Make canonical XML-like prompt and action tags single tokenizer tokens.
- End each successful user turn through the real `agent.finish` tool.
- Judge quality on raw generation only. Exact-match supervised lookup is not an
  accepted runtime or evaluation path.
- Keep `kjxlkj` integration API-first. `lkjai` should target typed resource
  routes instead of filesystem-shaped note workflows.

## Global Rules

1. Keep one canonical owner for each contract.
2. Keep each docs directory to one `README.md` plus multiple children.
3. Keep each docs file at `<= 300` lines.
4. Keep each authored source file at `<= 200` lines.
5. Prefer exact defaults, commands, paths, and payloads.
6. Remove conflicting legacy behavior instead of preserving it.
7. Docs-first workflow is mandatory: change docs, then code, then retrain.
8. Evaluation claims must match the real runtime path and actual tool execution.

## Top-Level Sections

- [vision/README.md](vision/README.md): product intent and LLM-first rules.
- [getting-started/README.md](getting-started/README.md): setup, run, verify.
- [product/README.md](product/README.md): chat, tools, and API behavior.
- [architecture/README.md](architecture/README.md): agent, runtime, model, training.
- [operations/README.md](operations/README.md): Compose, deployment, quality gates.
- [repository/README.md](repository/README.md): layout, workflow, and file rules.
- [research/README.md](research/README.md): external references that inform the canon.
- [decisions/README.md](decisions/README.md): accepted and rejected choices.

## Recommended Reading Order

1. [vision/purpose.md](vision/purpose.md)
2. [repository/workflow.md](repository/workflow.md)
3. [architecture/training/corpus.md](architecture/training/corpus.md)
4. [architecture/training/pipeline.md](architecture/training/pipeline.md)
5. [architecture/training/tokenizer.md](architecture/training/tokenizer.md)
6. [operations/training/agent-assessment.md](operations/training/agent-assessment.md)
7. [architecture/training/source-corpus.md](architecture/training/source-corpus.md)
8. [architecture/model/config.md](architecture/model/config.md)
9. [architecture/model/serving.md](architecture/model/serving.md)
10. [product/kjxlkj-integration.md](product/kjxlkj-integration.md)
11. [operations/training/competency-gate.md](operations/training/competency-gate.md)
