# Hardware Profiles

## Acceptance Gate

RTX 3070 8GB is the hard validation gate for accepted native CUDA work. A change
that only passes on larger or newer GPUs is a benchmark result, not an accepted
default.

The gate means:

- `docker compose --progress quiet --profile verify run --rm verify` must pass
  on RTX 3070.
- Focused 3070 native builds use
  `LKJAI_CUDA_ARCHS='86-real;86-virtual'`.
- Accepted dense reports must come from the dense BF16 CUDA path with
  `accepted_cuda_training=true`.
- Transformer reports, decode probes, CUDA Graph experiments, and NCCL tests are
  roadmap or diagnostics until their own CTest and report gates land.
- 40M start checks are compatibility diagnostics until a longer 40M run meets
  the documented promotion criteria.

## Benchmark Profile

RTX 5090/Blackwell is a higher-throughput profiling target. Use it to measure
headroom, memory pressure, cuBLASLt/cuDNN behavior, CUDA Graph readiness, and
future transformer kernels. Do not use it as the acceptance baseline.

Focused Blackwell profile builds use
`LKJAI_CUDA_ARCHS='120-real;120-virtual'`. The default native build also
includes Blackwell `120-real` and `120-virtual` flags.

When publishing a 5090 result, record it as a profile with:

- GPU name and compute capability,
- driver, CUDA toolkit, and cuDNN versions,
- exact train report fields and artifacts,
- whether the run is dense accepted, transformer experimental, decode
  diagnostic, or compatibility-only.

## Profile Table

| Profile | Role | Official facts | Repo policy |
|---|---|---|---|
| RTX 3070 | Acceptance gate | Ampere, 8 GB GDDR6, compute capability 8.6 | Must pass verify and dense acceptance on 8GB |
| RTX 5090 | Benchmark target | Blackwell, 32 GB GDDR7, compute capability 12.0 | Profile only; cannot relax the 3070 gate |

## Capability Notes

- CUDA BF16 requires compute capability 8.0 or higher.
- RTX 3070 compute capability 8.6 satisfies the BF16 requirement, so it remains
  the smallest supported acceptance profile.
- RTX 5090 compute capability 12.0 and larger memory make it useful for
  profiling future transformer and decode work.
- cuDNN SDPA supports FP16/BF16 attention inputs for eligible shapes; eligibility
  is still a runtime capability/report field, not a blanket acceptance claim.
- The new native profile configs are target/profile shapes, not accepted
  transformer CUDA training.

## Official References

- NVIDIA RTX 3070 reference specs:
  <https://www.nvidia.com/en-gb/geforce/graphics-cards/30-series/rtx-3070/>
- NVIDIA RTX 5090 reference specs:
  <https://www.nvidia.com/en-ph/geforce/graphics-cards/50-series/rtx-5090/>
- NVIDIA CUDA GPU compute capability table:
  <https://developer.nvidia.com/cuda/gpus>
- NVIDIA CUDA C++ floating-point type requirements:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>
- NVIDIA CUDA stream-ordered memory allocator:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html>
- NVIDIA cuDNN frontend attention docs:
  <https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html>
