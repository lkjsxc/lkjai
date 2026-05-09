# Documentation Canon

`docs/` is the only active canon for `lkjai`.

## System Goal

- Train and serve a commercial-safe scratch model through native C++/CUDA
  product binaries. The active default is about `40M` parameters for the
  current corpus; `60M` remains a later scale target.
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
9. Product training, serving, runtime, verification, and benchmark tooling must
   not depend on Rust or Python.

## Top-Level Sections

- [vision/README.md](vision/README.md): product intent and LLM-first rules.
- [current-state.md](current-state.md): accepted dense substrate, partial
  decoder limits, and the next decoder target.
- [getting-started/README.md](getting-started/README.md): setup, run, verify.
- [product/README.md](product/README.md): chat, tools, and API behavior.
- [architecture/README.md](architecture/README.md): agent, runtime, model, training.
- [operations/README.md](operations/README.md): Compose, deployment, performance,
  quality gates.
- [repository/README.md](repository/README.md): layout, workflow, and file rules.
- [research/README.md](research/README.md): external references that inform the canon.
- [decisions/README.md](decisions/README.md): accepted and rejected choices.

## Recommended Reading Order

1. [vision/purpose.md](vision/purpose.md)
2. [current-state.md](current-state.md)
3. [repository/workflow.md](repository/workflow.md)
4. [architecture/native/dense-substrate.md](architecture/native/dense-substrate.md)
5. [architecture/native/decoder/README.md](architecture/native/decoder/README.md)
6. [architecture/native/decoder/training.md](architecture/native/decoder/training.md)
7. [research/native-decoder-plan.md](research/native-decoder-plan.md)
8. [operations/performance/scale-profiles.md](operations/performance/scale-profiles.md)
9. [operations/performance/benchmarking.md](operations/performance/benchmarking.md)
10. [architecture/training/corpus.md](architecture/training/corpus.md)
11. [architecture/training/pipeline.md](architecture/training/pipeline.md)
12. [architecture/training/tokenizer.md](architecture/training/tokenizer.md)
13. [operations/training/agent-assessment.md](operations/training/agent-assessment.md)
14. [architecture/training/source-corpus.md](architecture/training/source-corpus.md)
15. [architecture/model/config.md](architecture/model/config.md)
16. [architecture/native/strategy.md](architecture/native/strategy.md)
17. [architecture/native/cuda-stack.md](architecture/native/cuda-stack.md)
18. [architecture/native/implementation-backlog.md](architecture/native/implementation-backlog.md)
19. [operations/performance/hardware-profiles.md](operations/performance/hardware-profiles.md)
20. [architecture/model/serving.md](architecture/model/serving.md)
21. [product/kjxlkj-integration.md](product/kjxlkj-integration.md)
22. [operations/training/competency-gate.md](operations/training/competency-gate.md)
