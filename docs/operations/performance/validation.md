# CUDA Validation Matrix

## Purpose

Native CUDA work is accepted by parity, stability, and runtime-contract tests,
not by finite loss alone.

## Required Checks

| Area | Check |
|---|---|
| Device substrate | allocation, free, shape, dtype, and host-device round trip |
| Capability | CC, BF16, cuBLASLt, cuDNN, and SDPA eligibility JSON |
| Forward | tiny-model logits parity against CPU FP32 reference |
| Attention | MHA, GQA, masks, and sequence-length parity |
| Backward | finite-difference checks on small layers |
| Optimizer | one optimizer step decreases or preserves tiny-batch loss trend |
| Resume | restart equivalence for counters, LR, optimizer state, and checksums |
| Export | load/save round trip and tokenizer/config checksum match |
| Server | `/v1/models`, unsupported decode, and later real decode contracts |

## Metrics

Benchmark reports must include:

- training tokens/sec,
- p50 and p95 microstep latency,
- batch load, H2D, forward, backward, optimizer, and checkpoint timing,
- peak and steady device memory,
- prefill tokens/sec after decode lands,
- decode ms/token at batch `1`, `4`, and `8` after decode lands.

## Failure Rule

If a faster CUDA path fails parity, the reference path remains canonical and the
faster path stays disabled behind an explicit backend flag.
