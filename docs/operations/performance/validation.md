# CUDA Validation Matrix

## Purpose

Native CUDA work is accepted by parity, stability, and runtime-contract tests,
not by finite loss alone.

The canonical hardware gate is RTX 3070 8GB. Larger GPUs, including
RTX 5090/Blackwell, are benchmark profiles unless the same change also passes
the RTX 3070 gate.

## Required Checks

| Area | Check |
|---|---|
| Device substrate | allocation, free, shape, dtype, and host-device round trip |
| Capability | CC, BF16, cuBLASLt, cuDNN, and SDPA eligibility JSON |
| Build policy | default CUDA arch list includes Blackwell `120` and accepts focused env overrides |
| Config policy | every native/training config uses known keys and valid BF16 dimensions |
| Forward | tiny-model logits parity against CPU FP32 reference |
| Attention | MHA, GQA, masks, and sequence-length parity |
| Backward | finite-difference checks on small layers |
| Optimizer | one optimizer step decreases or preserves tiny-batch loss trend |
| Resume | restart equivalence for counters, LR, optimizer state, and checksums |
| Export | load/save round trip, tokenizer/config checksum match, and dense BF16 export logits parity against FP32 checkpoint masters |
| Server | `/v1/models`, unsupported decode, and later real decode contracts |

## Canonical Verify

```bash
docker compose --progress quiet --profile verify run --rm verify
```

The older `up --abort-on-container-exit verify` form is acceptable for
interactive diagnosis, but the `run --rm verify` form is the canonical gate.

## Foundation CTest Gates

- `native_config_contract` passes only when native/training configs use known
  keys, BF16 dtype, valid vocab/context bounds, valid `heads * head_dim`, valid
  `heads % kv_heads`, and existing repo-local native config references.
- `native_cuda_arch_contract` passes only when CMake, Docker, and Compose keep
  `LKJAI_CUDA_ARCHS` support and the default CMake policy still includes
  Blackwell `120-real` and `120-virtual`.
- `native_dense_cuda_parity` now wraps `lkjai-native-dense-check` and fails if
  dense CUDA parity or additive hardware/build capability fields are missing.

## Metrics

Benchmark reports must include:

- training tokens/sec,
- p50 and p95 microstep latency,
- batch load, H2D, forward, backward, optimizer, checkpoint, and export timing,
- peak and steady device memory,
- prefill tokens/sec after decode lands,
- decode ms/token at batch `1`, `4`, and `8` after decode lands.

## Failure Rule

If a faster CUDA path fails parity, the reference path remains canonical and the
faster path stays disabled behind an explicit backend flag.
