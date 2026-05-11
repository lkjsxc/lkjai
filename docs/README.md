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
  [operations/performance/profiles/scale-profiles.md](operations/performance/profiles/scale-profiles.md).
- Runtime and API: [architecture/native/contracts/runtime.md](architecture/native/contracts/runtime.md),
  [architecture/runtime/inference.md](architecture/runtime/inference.md),
  [product/api.md](product/api.md), [product/chat.md](product/chat.md),
  [product/kjxlkj-integration.md](product/kjxlkj-integration.md).
- Corpus and tokenizer: [architecture/training/data/corpus.md](architecture/training/data/corpus.md),
  [architecture/training/data/source-corpus.md](architecture/training/data/source-corpus.md),
  [architecture/training/pipeline/pipeline.md](architecture/training/pipeline/pipeline.md),
  [architecture/training/data/tokenizer.md](architecture/training/data/tokenizer.md).
- Performance evidence: [operations/performance/measurement/benchmarking.md](operations/performance/measurement/benchmarking.md),
  [operations/performance/evidence/evidence.md](operations/performance/evidence/evidence.md),
  [operations/performance/profiles/hardware-profiles.md](operations/performance/profiles/hardware-profiles.md),
  [operations/performance/contracts/benchmark-output.md](operations/performance/contracts/benchmark-output.md).
- Verification: [getting-started/verification.md](getting-started/verification.md),
  [operations/quality.md](operations/quality.md),
  [operations/compose.md](operations/compose.md),
  [repository/workflow.md](repository/workflow.md).
- Adding report fields: [architecture/native/contracts/training.md](architecture/native/contracts/training.md),
  [architecture/native/contracts/contract-inventory.md](architecture/native/contracts/contract-inventory.md),
  [operations/performance/contracts/train-report-fields.md](operations/performance/contracts/train-report-fields.md),
  [operations/performance/measurement/benchmarking.md](operations/performance/measurement/benchmarking.md).

## Contract Owner Index

- Current state and claim limits: [current-state.md](current-state.md).
- Native contract inventory: [architecture/native/contracts/contract-inventory.md](architecture/native/contracts/contract-inventory.md).
- Decoder acceptance: [architecture/native/decoder/training.md](architecture/native/decoder/training.md).
- Decoder config shape: [architecture/native/decoder/config.md](architecture/native/decoder/config.md).
- Decoder KV-cache decode: [architecture/native/decoder/kv-cache.md](architecture/native/decoder/kv-cache.md).
- Train reports: [architecture/native/contracts/training.md](architecture/native/contracts/training.md).
- Packed cache and dataset lineage:
  [architecture/training/data/packed-cache.md](architecture/training/data/packed-cache.md),
  [architecture/training/data/dataset.md](architecture/training/data/dataset.md),
  [architecture/training/data/provenance.md](architecture/training/data/provenance.md).
- Benchmark evidence: [operations/performance/measurement/benchmarking.md](operations/performance/measurement/benchmarking.md)
  and [operations/performance/evidence/evidence.md](operations/performance/evidence/evidence.md).
- Benchmark output fields and artifacts:
  [operations/performance/contracts/benchmark-output.md](operations/performance/contracts/benchmark-output.md),
  [operations/performance/contracts/train-report-fields.md](operations/performance/contracts/train-report-fields.md),
  [operations/performance/contracts/benchmark-artifacts.md](operations/performance/contracts/benchmark-artifacts.md),
  [operations/performance/contracts/promotion-criteria.md](operations/performance/contracts/promotion-criteria.md).
- Compose and verification: [operations/compose.md](operations/compose.md),
  [operations/quality.md](operations/quality.md).
