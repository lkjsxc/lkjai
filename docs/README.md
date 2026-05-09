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

## Task Routes

- Decoder training: [current-state.md](current-state.md),
  [architecture/native/decoder/README.md](architecture/native/decoder/README.md),
  [architecture/native/decoder/training.md](architecture/native/decoder/training.md),
  [architecture/native/decoder/config.md](architecture/native/decoder/config.md),
  [operations/performance/scale-profiles.md](operations/performance/scale-profiles.md).
- Runtime and API: [architecture/native/runtime.md](architecture/native/runtime.md),
  [architecture/runtime/inference.md](architecture/runtime/inference.md),
  [product/api.md](product/api.md), [product/chat.md](product/chat.md),
  [product/kjxlkj-integration.md](product/kjxlkj-integration.md).
- Corpus and tokenizer: [architecture/training/corpus.md](architecture/training/corpus.md),
  [architecture/training/source-corpus.md](architecture/training/source-corpus.md),
  [architecture/training/pipeline.md](architecture/training/pipeline.md),
  [architecture/training/tokenizer.md](architecture/training/tokenizer.md).
- Performance evidence: [operations/performance/benchmarking.md](operations/performance/benchmarking.md),
  [operations/performance/evidence.md](operations/performance/evidence.md),
  [operations/performance/hardware-profiles.md](operations/performance/hardware-profiles.md),
  [operations/performance/benchmark-output.md](operations/performance/benchmark-output.md).
- Verification: [getting-started/verification.md](getting-started/verification.md),
  [operations/quality.md](operations/quality.md),
  [operations/compose.md](operations/compose.md),
  [repository/workflow.md](repository/workflow.md).
- Adding report fields: [architecture/native/training.md](architecture/native/training.md),
  [architecture/native/contract-inventory.md](architecture/native/contract-inventory.md),
  [operations/performance/benchmark-output.md](operations/performance/benchmark-output.md),
  [operations/performance/benchmarking.md](operations/performance/benchmarking.md).

## Contract Owner Index

- Current state and claim limits: [current-state.md](current-state.md).
- Native contract inventory: [architecture/native/contract-inventory.md](architecture/native/contract-inventory.md).
- Decoder acceptance: [architecture/native/decoder/training.md](architecture/native/decoder/training.md).
- Decoder config shape: [architecture/native/decoder/config.md](architecture/native/decoder/config.md).
- Decoder KV-cache decode: [architecture/native/decoder/kv-cache.md](architecture/native/decoder/kv-cache.md).
- Train reports: [architecture/native/training.md](architecture/native/training.md).
- Packed cache and dataset lineage:
  [architecture/training/packed-cache.md](architecture/training/packed-cache.md),
  [architecture/training/dataset.md](architecture/training/dataset.md),
  [architecture/training/provenance.md](architecture/training/provenance.md).
- Benchmark evidence: [operations/performance/benchmarking.md](operations/performance/benchmarking.md)
  and [operations/performance/evidence.md](operations/performance/evidence.md).
- Compose and verification: [operations/compose.md](operations/compose.md),
  [operations/quality.md](operations/quality.md).
