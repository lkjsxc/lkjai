# Hardware Profiles

## Acceptance Gate

RTX 3070 is the hard validation gate for accepted native CUDA work. A change
that only passes on larger or newer GPUs is a benchmark result, not an accepted
default. NVIDIA's CUDA GPU table lists RTX 3070 under compute capability 8.6.

The gate means:

- `docker compose --progress quiet --profile verify run --build --rm verify` must pass
  on RTX 3070.
- Focused 3070 native builds use
  `LKJAI_CUDA_ARCHS='86-real;86-virtual'`.
- Accepted dense reports must come from the dense BF16 CUDA path with
  `accepted_cuda_training=true`.
- Transformer reports, decode probes, CUDA Graph experiments, and NCCL tests are
  roadmap or diagnostics until their own CTest and report gates land.
- 40M start checks are compatibility diagnostics until a longer 40M run meets
  the documented promotion criteria.

## Benchmark Profiles

RTX 4090/Ada and RTX 5090/Blackwell are higher-throughput profiling targets.
Use them to measure headroom, memory pressure, cuBLASLt/cuDNN behavior, CUDA
Graph readiness, and future transformer kernels. Do not use them as the
acceptance baseline.

Focused Blackwell profile builds use
`LKJAI_CUDA_ARCHS='120-real;120-virtual'`. The default native build also
includes Blackwell `120-real` and `120-virtual` flags.

Focused Ada profile builds may use `LKJAI_CUDA_ARCHS='89-real;89-virtual'`.
The default native build also includes Ada `89-real` and `89-virtual` flags.

When publishing a 4090, 5090, or generic recent NVIDIA GPU result, record it as
a profile with:

- GPU name and compute capability,
- driver, CUDA toolkit, and cuDNN builds,
- exact train report fields and artifacts,
- whether the run is dense accepted, transformer experimental, decode
  diagnostic, or compatibility-only.

## Profile Table

| Profile | Role | Official facts | Repo policy |
|---|---|---|---|
| RTX 3070 | Acceptance gate | CUDA table lists compute capability 8.6 | Must pass verify and dense acceptance |
| RTX 4090/Ada | Benchmark target | CUDA table lists RTX 4090 and Ada GPUs under compute capability 8.9 | Profile only; cannot relax the 3070 gate |
| RTX 5090/Blackwell | Benchmark target | CUDA table lists RTX 5090 under compute capability 12.0 | Profile only; cannot relax the 3070 gate |
| Recent NVIDIA GPU | Diagnostic target | Record the CUDA table compute capability and report fields | Profile only unless separately accepted |

## Capability Notes

- RTX 3070 compute capability 8.6 remains the acceptance profile for dense
  BF16 CUDA work.
- RTX 4090/Ada compute capability 8.9 and RTX 5090/Blackwell compute
  capability 12.0 are profile targets for future transformer and decode work.
- Blackwell tuning follows NVIDIA's Blackwell guide: start from general CUDA
  best practices, then tune architecture-specific occupancy, memory, and launch
  behavior after correctness gates pass.
- cuDNN provides tuned primitives for attention, matmul, pooling, convolution,
  and normalization. SDPA eligibility is a runtime capability/report field, not
  a blanket acceptance claim.
- Stream-ordered allocation remains guarded by the runtime capability field
  because CUDA exposes device support through memory-pool attributes.
- The new native profile configs are target/profile shapes, not accepted
  transformer CUDA training.

## Official References

- CUDA 12.8 release notes:
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html>
- CUDA GPU compute capability:
  <https://developer.nvidia.com/cuda-gpus>
- Blackwell tuning guide:
  <https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-tuning-guide/index.html>
- CUDA stream-ordered allocator:
  <https://docs.nvidia.com/cuda/archive/13.1.2/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html>
- cuDNN documentation:
  <https://docs.nvidia.com/cudnn/index.html>
