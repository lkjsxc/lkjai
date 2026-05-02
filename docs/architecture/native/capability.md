# Native Capability JSON

Native capability reporting is one reusable API consumed by dense checks,
server health/model responses, training reports, and benchmark records.

## Shape

Capability JSON includes:

- `cuda_available`: whether a CUDA device can be selected.
- `device`: `cuda` or `cpu`.
- `gpu_name`: CUDA device name when available.
- `compute_capability`: `[major, minor]`.
- `cuda_runtime_version`: integer CUDA runtime version.
- `cudnn_version`: integer cuDNN runtime version.
- `bf16_supported`: compute capability `8.0+`.
- `cublaslt_available`: cuBLASLt handle creation succeeded.
- `cudnn_available`: cuDNN handle creation succeeded.
- `sdpa_eligible`: BF16 plus a shape/library gate that permits cuDNN SDPA.
- `async_alloc_supported`: CUDA memory-pool allocation is usable.
- `warning`: human-readable degraded-mode reason.
- `error`: hard failure reason for capability check executables.

## Consumers

- `lkjai-native-dense-check` exits non-zero when required CUDA/BF16 capability
  is missing.
- `/healthz` reports capability even when the artifact is missing.
- `/v1/models` reports capability only after the model artifact is loadable.
- Training reports embed the same fields so performance results can be compared
  across hosts.

## Policy

CPU mode is diagnostic only for this native CUDA roadmap. It must be visible in
JSON and must not count as dense-forward, attention, backward, or decode
acceptance.
