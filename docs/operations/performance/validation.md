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
| Forward | tiny-model logits parity against CPU FP32 reference |
| Attention | MHA, GQA, masks, and sequence-length parity |
| Backward | finite-difference checks on small layers |
| Optimizer | one optimizer step decreases or preserves tiny-batch loss trend |
| Resume | restart equivalence for counters, LR, optimizer state, and checksums |
| Export | load/save round trip, tokenizer/config checksum match, and dense BF16 export logits parity against FP32 checkpoint masters |
| Server | `/v1/models`, unsupported decode, and later real decode contracts |

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
