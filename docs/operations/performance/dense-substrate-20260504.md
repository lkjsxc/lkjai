# Dense Substrate Evidence 2026-05-04

This record captures the RTX 3070 evidence for the dense CUDA substrate
tuning patch set. Raw artifacts are intentionally left under ignored
`artifacts/` and `data/perf-runs/`; this file keeps only the curated facts.

## Patch Set

- `native: add dense tuning report fields`
- `docs: define dense cuda substrate tuning`
- `native: move source lists to cmake fragment`
- `docs: fix runtime tuning toc links`

The implementation keeps train report `schema_version=3` and adds optional
fields for dense autotune, workspace, allocator, timing, and LM-head cache
state.

## Hardware

- GPU: NVIDIA GeForce RTX 3070
- Compute capability: 8.6
- Device memory: 8192 MiB
- Acceptance role: 8 GB RTX 3070 gate

## Verification

```bash
docker build -f ops/docker/Dockerfile.native --target build \
  -t lkjai-native-build-check:local .

docker compose --progress quiet --profile verify run --build --rm verify
```

Both commands passed after the patch set.

## Dense Learning Control

Baseline command family:

```bash
lkjai-native-train --train --mode dense --max-steps 1024
```

Post-change command family:

```bash
lkjai-native-train --train --mode dense --max-steps 1024
```

| Field | Baseline | Post-change |
|---|---:|---:|
| tokens/sec | 104184 | 129729 |
| throughput ratio | 1.0 | 1.24519 |
| backward seconds | 0.0222777 | 0.0217617 |
| backward speedup | 1.0 | 1.02371 |
| checkpoint checksum | `e5e816f19fa66b8b` | `e5e816f19fa66b8b` |

The repo comparator reported the same speed numbers but did not mark the pair
as accepted because `dataset_digest` differs. That is expected for these
run IDs because the synthetic packed-cache metadata is run-specific. Treat
this pair as strong local evidence, not promoted benchmark evidence.

## Active Runtime Fields

The post-change report confirms the new dense runtime path was active:

- `timing_source`: `cuda_events_deferred_slot_sync`
- `dense_autotune_enabled`: `true`
- `dense_autotune_mode`: `heuristic`
- `dense_workspace_sweep_bytes`: `4194304`
- `cublaslt_workspace_bytes`: `4194304`
- `dense_allocator_backend`: `cuda_malloc_async_pool`
- `dense_timing_mode`: `deferred`
- `dense_head_f32_cache_enabled`: `true`
- `dense_head_f32_cache_refreshes`: `1024`

Transient dense buffers in the debug learning-control run were:

- `dense_step_logits_bytes`: `65536`
- `dense_step_grad_logits_bytes`: `65536`
- `dense_step_d_hidden_bytes`: `8192`

## Bounded 40M Check

Command:

```bash
docker compose --profile train run --rm train \
  --train --mode dense \
  --config /workspace/configs/native/native_40m_bf16.json \
  --packed-cache /app/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --seq-len 1024 --max-steps 4
```

Result:

- status: `success`
- promotion status: `compatibility_only`
- tokens/sec: `1449.45`
- initial loss: `9.01055`
- logits reference: `pass`
- logits max absolute difference: `8.39517e-05`
- logits tolerance: `0.01`
- checkpoint checksum: `6aed937ce45a60f5`

The compatibility run intentionally is not promotable. It is a bounded start
check for the 40M shape and does not require loss improvement.

40M transient dense buffers were:

- `dense_step_logits_bytes`: `33554432`
- `dense_step_grad_logits_bytes`: `33554432`
- `dense_step_d_hidden_bytes`: `2359296`
- `cublaslt_workspace_bytes`: `4194304`

## Limitations

This record does not claim transformer CUDA training, native autoregressive
decode, CUDA Graph replay, NCCL scaling, TensorRT integration, or FP16
fallback support. The accepted implementation scope remains the dense BF16
CUDA milestone.
