# Native Contract Inventory

## Purpose

This inventory is the canonical list of stable native surfaces for foundation
work. Changes here must preserve current dense CUDA acceptance while making
future transformer CUDA work measurable.

## Stable Surfaces

- Packed cache format: `lkjai-packed-cache-v2`.
- Packed cache files: `metadata.json`, `tokens.bin`, `loss_mask.bin`, and
  `starts.bin`.
- Artifact format: `lkjai-native-artifact-v2`.
- Artifact kinds: `export` and `checkpoint`.
- Artifact model kinds: `dense` and `transformer`.
- Train report schema: `schema_version=3`.
- Native runtime HTTP boundary: `/api/chat`, `/api/model`, and `/api/runs/{id}`.
- Native HTTP boundary: `/v1/models` and `/v1/chat/completions`.
- Native health boundary: `/healthz`.

## Current Acceptance

- Dense BF16 CUDA embedding plus LM-head training is accepted when the report
  says `accepted_cuda_training=true`, `implementation_status=accepted`, and
  `dense_cuda_path=true`.
- Transformer training is experimental host/reference plumbing. Reports must
  say `accepted_cuda_training=false`, `implementation_status=experimental`, and
  `transformer_cuda_path=false`.
- Native chat decode is unsupported for dense and transformer artifacts.
  `/v1/chat/completions` returns HTTP `422` with no `choices` for those kinds.
- Decoder artifacts may return `choices` through the host/reference decode
  bridge, but accepted decoder CUDA training and accepted KV-cache decode are
  still future gates.
- The 40M shape is compatibility and profiling only until a long run satisfies
  the documented promotion criteria.

## Additive Surfaces

- `LKJAI_CUDA_ARCHS` may select native CUDA architecture flags for CMake and
  Docker builds.
- Capability JSON may add hardware and build fields such as driver version,
  device count, selected device index, total global memory, SM count, CUDA
  architecture flags, and async allocation support.
- Train reports may embed the same additive capability fields while remaining
  schema version `3`.
- Profile configs may be added for RTX 3070 and RTX 5090/Blackwell targets, but
  they do not promote transformer CUDA training.

## Diagnostic Surfaces

- CPU mode is diagnostic only and must stay visible in health/model JSON.
- `sdpa_eligible` is a device/library-level signal in this foundation phase.
  Shape-specific cuDNN SDPA acceptance belongs to the transformer-forward phase.
- RTX 5090/Blackwell reports are profiling data unless the RTX 3070 acceptance
  gate also passes.

## Future-Versioned Surfaces

- Accepted transformer CUDA training may require new report fields, but schema
  `3` remains stable until a concrete transformer-forward/backward gate proves a
  schema break is necessary.
- Accepted autoregressive decode, KV cache layout, CUDA Graph capture, NCCL,
  and activation checkpointing are future milestones and are not current
  accepted capabilities.
