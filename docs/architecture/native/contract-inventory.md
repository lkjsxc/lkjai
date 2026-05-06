# Native Contract Inventory

Owner: `docs/architecture/native/contract-inventory.md`.
State: canonical.

## Purpose

This inventory is the canonical list of stable native surfaces for foundation
work. Changes here must preserve current dense CUDA acceptance while making
future transformer CUDA work measurable.

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

- contract_id: `packed-cache-v2`.
- Owner: `docs/architecture/training/packed-cache.md`.
- Surface: `lkjai-packed-cache-v2` with `metadata.json`, `tokens.bin`,
  `loss_mask.bin`, and `starts.bin`.
- Status: `accepted`.
- Acceptance: accepted native training consumes this format as the training
  input path.
- Non-claims: source ingestion and tokenizer training are separate pipeline
  contracts.

### Native Artifact

- contract_id: `native-artifact-v2`.
- Owner: `docs/architecture/native/artifact.md`.
- Surface: `lkjai-native-artifact-v2`; kinds `export` and `checkpoint`; model
  kinds `dense`, `decoder`, and `transformer`.
- Status: `accepted` for dense export/checkpoint, `partial` for decoder,
  `experimental` for transformer.
- Acceptance: artifact inspect, logits check, checksum, and server load pass for
  the claimed model kind.
- Non-claims: artifact load does not imply chat-capable decode.

### Train Report

- contract_id: `train-report-v3`.
- Owner: `docs/architecture/native/training.md`.
- Surface: `train-report.json`, `schema_version=3`.
- Status: `accepted`.
- Acceptance: every train or smoke run writes the stable schema and additive
  backend/capability fields without changing schema version.
- Non-claims: additive fields do not promote experimental or partial runs.

### Native HTTP Runtime

- contract_id: `native-http-runtime`.
- Owner: `docs/architecture/native/runtime.md`.
- Surface: `/healthz`, `/v1/models`, `/v1/chat/completions`, `/api/chat`,
  `/api/model`, `/api/config`, and `/api/runs/{id}`.
- Status: `accepted` for route shape and unsupported-decode behavior.
- Acceptance: native server loads readable artifacts and reports unsupported
  decode truthfully for unsupported model kinds.
- Non-claims: route availability does not imply accepted low-latency decode.

### Dense Foundation

- contract_id: `dense-foundation`.
- Owner: `docs/architecture/native/dense-decoder.md`.
- Surface: dense BF16 CUDA embeddings plus LM-head training, FP32 AdamW state,
  packed-cache input, checkpoint/export, and logits check.
- Status: `accepted`.
- Acceptance: reports say `accepted_cuda_training=true`,
  `implementation_status=accepted`, and `dense_cuda_path=true`.
- Non-claims: dense evidence does not prove RoPE, RMSNorm, GQA, SwiGLU, tied
  embeddings, block backward, or KV-cache decode.

### First Decoder Acceptance Target

- contract_id: `decoder-40m-3070-acceptance`.
- Owner: `docs/architecture/native/decoder/training.md`.
- Surface: `configs/native/decoder_40m_bf16_3070.json` with
  `configs/training/decoder_2h_40m_3070.json`.
- Status: `future`.
- Acceptance: one RTX 3070 8GB run reports
  `implementation_status=accepted`, `accepted_cuda_training=true`,
  `decoder_cuda_slice=full_decoder`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and
  `decode_backend=cuda_kv_cache`.
- Non-claims: `native_dense_40m_bf16_3070.json` remains dense compatibility and
  profile evidence, not same-model decoder acceptance.

### Decoder Decode

- contract_id: `decoder-kv-cache-decode`.
- Owner: `docs/architecture/native/decoder/decode.md`.
- Surface: autoregressive decoder chat through `/v1/chat/completions` and
  `/api/chat`.
- Status: `partial`.
- Acceptance: accepted decode requires native BF16 KV cache, no per-token device
  allocation in steady state, and real chat `choices`.
- Non-claims: `host_reference_recompute` with `kv_cache_backend=none` is not
  accepted serving evidence.

### Transformer Lane

- contract_id: `transformer-reference-lane`.
- Owner: `docs/architecture/native/transformer-cuda-roadmap.md`.
- Surface: retained transformer host/reference and probe paths.
- Status: `experimental`.
- Acceptance: reports must say `accepted_cuda_training=false`,
  `implementation_status=experimental`, and `transformer_cuda_path=false`.
- Non-claims: transformer diagnostics do not change dense or decoder
  acceptance.

### Hardware And Capability Fields

- contract_id: `hardware-capability-fields`.
- Owner: `docs/operations/performance/hardware-profiles.md`.
- Surface: `LKJAI_CUDA_ARCHS`, capability JSON, and additive train-report
  hardware/build fields.
- Status: `additive`.
- Acceptance: reports may include driver version, device count, selected device,
  total memory, SM count, CUDA architecture flags, async allocation support,
  cuBLASLt, cuDNN, and SDPA eligibility while staying schema version `3`.
- Non-claims: `sdpa_eligible` is device/library-level until shape-specific
  parity and timing are proven.

### Backend Roadmap

- contract_id: `native-decoder-backend-roadmap`.
- Owner: `docs/research/native-decoder-roadmap.md`.
- Surface: backend priority and later scale features.
- Status: `future`.
- Acceptance: cuBLASLt is the default GEMM owner; first decoder acceptance may
  use `attention_backend=cuda_causal_gqa_bf16_reference`; cuDNN SDPA is the
  later attention performance backend after parity.
- Non-claims: CUTLASS, CUDA Graphs, TensorRT, TensorRT-LLM, NCCL, and
  activation checkpointing are not first acceptance requirements.
