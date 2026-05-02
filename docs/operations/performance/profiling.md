# Profiling Protocol

## Goal

Keep native CUDA optimization evidence-based.

## Required Ranges

Native binaries annotate these coarse scopes when profiling is enabled:

- `loader`
- `h2d`
- `forward`
- `attention`
- `backward`
- `optimizer`
- `decode_step`
- `http_request`

Do not annotate tiny inner fragments until a profiler trace proves they matter.

## Nsight Systems

Use system timelines to decide whether wall time is CPU loader, H2D copy,
kernel launch, GPU compute, or HTTP overhead.

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=nvtx \
  --capture-range-end=stop \
  -o artifacts/profiles/train_nsys \
  lkjai-native-train --train
```

## Nsight Compute

Use kernel analysis only after the timeline identifies a specific kernel or
library call.

```bash
ncu \
  --set roofline \
  --target-processes all \
  --kernel-name-base demangled \
  -o artifacts/profiles/train_ncu \
  lkjai-native-train --train
```

## Acceptance

- A speed claim links to the benchmark summary and the profiling run id.
- A custom kernel must beat the vendor-library baseline for the active shape.
- CUDA Graph replay is enabled only after shapes and launch order are stable.
