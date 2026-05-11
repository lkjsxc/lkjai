# Native Contract Inventory

Owner: `docs/architecture/native/contracts/contract-inventory.md`.
State: canonical.

## Purpose

This inventory is the canonical list of stable native surfaces for foundation
work. Changes here must preserve current dense CUDA acceptance while making
decoder product acceptance measurable.

## Ownership Fields

Each contract record uses these fields:

- `contract_id`: stable machine-readable name.
- `owner`: canonical detail document for the rule.
- `state`: `accepted`, `experimental`, `partial`, `future`, or `additive`.
- `canonical_source`: exact schema, route, config, or runtime boundary.
- `supersedes`: legacy or conflicting rule removed by this contract.
- `acceptance`: report fields, commands, or observable behavior required.
- `non_claims`: behavior this contract does not prove.

## Contract Records

### Packed Cache

- contract_id: `packed-cache`.
- owner: `docs/architecture/training/data/packed-cache.md`.
- canonical_source: `lkjai-packed-cache` with `metadata.json`, `tokens.bin`,
  `loss_mask.bin`, and `starts.bin`.
- state: `accepted`.
- supersedes: none.
- acceptance: accepted native training consumes this format as the training
  input path.
- non_claims: source ingestion and tokenizer training are separate pipeline
  contracts.

### Native Artifact

- contract_id: `native-artifact`.
- owner: `docs/architecture/native/contracts/artifact.md`.
- canonical_source: `lkjai-native-artifact`; kinds `export` and `checkpoint`; model
  kinds `dense`, `decoder`, and `transformer`.
- state: `accepted` for dense export/checkpoint, `partial` for decoder,
  `experimental` for transformer.
- supersedes: none.
- acceptance: artifact inspect, logits check, checksum, and server load pass for
  the claimed model kind.
- non_claims: artifact load does not imply accepted chat decode.

### Train Report

- contract_id: `train-report`.
- owner: `docs/architecture/native/contracts/training.md`.
- canonical_source: `train-report.json`, `schema=lkjai-train-report`.
- state: `accepted`.
- supersedes: none.
- acceptance: every train or smoke run writes the stable schema and additive
  backend/capability fields without changing schema identifier.
- non_claims: additive fields do not promote experimental or partial runs.

### Native HTTP Runtime

- contract_id: `native-http-runtime`.
- owner: `docs/architecture/native/contracts/runtime.md`.
- canonical_source: `/healthz`, `/v1/models`, `/v1/chat/completions`,
  `/api/dense/status`, `/api/dense/next-token`, `/api/chat`, `/api/model`,
  `/api/config`, and `/api/runs/{id}`.
- state: `accepted` for route shape and unsupported-decode behavior.
- supersedes: none.
- acceptance: native server loads readable artifacts and reports unsupported
  decode truthfully for unsupported model kinds.
- non_claims: route availability does not imply accepted low-latency decode.

### Dense Browser Demo

- contract_id: `dense-browser-demo`.
- owner: `docs/product/dense-demo.md`.
- canonical_source: `GET /`, `GET /api/dense/status`, and
  `POST /api/dense/next-token`.
- state: `additive`.
- supersedes: chat-first root page for dense artifacts.
- acceptance: browser page and route contract tests prove dense status,
  top-k logits, checksum, and unsupported chat disclosure.
- non_claims: dense demo evidence does not prove autoregressive chat.

### Dense Foundation

- contract_id: `dense-foundation`.
- owner: `docs/architecture/native/dense/decoder.md`.
- canonical_source: dense BF16 CUDA embeddings plus LM-head training, FP32 AdamW state,
  packed-cache input, checkpoint/export, and logits check.
- state: `accepted`.
- supersedes: none.
- acceptance: reports say `accepted_cuda_training=true`,
  `implementation_status=accepted`, and `dense_cuda_path=true`.
- non_claims: dense evidence does not prove RoPE, RMSNorm, GQA, SwiGLU, tied
  embeddings, block backward, or KV-cache decode.

### First Decoder Acceptance Target

- contract_id: `decoder-40m-3070-acceptance`.
- owner: `docs/architecture/native/decoder/training.md`.
- canonical_source: `configs/native/decoder_40m_bf16_3070.json` with
  `configs/training/decoder_2h_40m_3070.json`.
- state: `future`.
- supersedes: none.
- acceptance: governed by
  [decoder/training.md](../decoder/training.md), including full-decoder CUDA
  training, logits/export/server evidence, block-weight deltas, and accepted
  KV-cache decode names.
- non_claims: current dirty decoder scaffolding, synthetic block gradients,
  host recompute decode, and `native_dense_40m_bf16_3070.json` are not
  same-model decoder acceptance.

### Decoder Decode

- contract_id: `decoder-kv-cache-decode`.
- owner: `docs/architecture/native/decoder/decode.md`.
- canonical_source: autoregressive decoder chat through `/v1/chat/completions` and
  `/api/chat`.
- state: `partial`.
- supersedes: none.
- acceptance: governed by [decoder/training.md](../decoder/training.md) and
  [decoder/decode.md](../decoder/decode.md); accepted decode requires native
  CUDA BF16 KV cache consumed by generation, no per-token device allocation in
  steady state, and real chat `choices`.
- non_claims: `host_reference_recompute` with
  `kv_cache_backend=host_contiguous_bf16_diagnostic` is not accepted serving
  evidence.

### Transformer Lane

- contract_id: `transformer-reference-lane`.
- owner: `docs/architecture/native/cuda/transformer-cuda-plan.md`.
- canonical_source: retained transformer host/reference and probe paths.
- state: `experimental`.
- supersedes: none.
- acceptance: reports must say `accepted_cuda_training=false`,
  `implementation_status=experimental`, and `transformer_cuda_path=false`.
- non_claims: transformer diagnostics do not change dense or decoder
  acceptance.

### Hardware And Capability Fields

- contract_id: `hardware-capability-fields`.
- owner: `docs/operations/performance/profiles/hardware-profiles.md`.
- canonical_source: `LKJAI_CUDA_ARCHS`, capability JSON, and additive train-report
  hardware/build fields.
- state: `additive`.
- supersedes: none.
- acceptance: reports may include driver build number, device count, selected device,
  total memory, SM count, CUDA architecture flags, async allocation support,
  cuBLASLt, cuDNN, and SDPA eligibility while staying stable schema.
- non_claims: `sdpa_eligible` is device/library-level until shape-specific
  parity and timing are proven.

### Backend Plan

- contract_id: `native-decoder-backend-backlog`.
- owner: `docs/research/native-decoder-plan.md`.
- canonical_source: backend priority and later scale features.
- state: `future`.
- supersedes: none.
- acceptance: cuBLASLt is the default GEMM owner; first decoder acceptance may
  use `attention_backend=cuda_causal_gqa_bf16_reference`; cuDNN SDPA is the
  later attention performance backend after parity.
- non_claims: CUTLASS, CUDA Graphs, TensorRT, TensorRT-LLM, NCCL, and
  activation checkpointing are not first acceptance requirements.
