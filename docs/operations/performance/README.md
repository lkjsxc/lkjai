# Performance

This subtree is the canonical performance contract for `lkjai`.

## Purpose

- Maximize training throughput for the active `scratch-40m` model contract.
- Keep optimization decisions measurable and reproducible.
- Prefer repo-local, Compose-runnable workflows over one-off host commands.

## Contents

- [training-speed.md](training-speed.md): current bottlenecks, target state,
  and optimization order.
- [benchmarking.md](benchmarking.md): required measurements, artifacts, and
  acceptance workflow.
- [profiling.md](profiling.md): Nsight and NVTX protocol for native work.
- [kernel-roadmap.md](kernel-roadmap.md): library, compiler, Triton, and CUDA
  escalation order.
- [validation.md](validation.md): numerical, resume, and server acceptance
  matrix for CUDA work.

## Active Priority

Training speed is the first priority. Inference improvements are accepted when
they share the same model/cache foundations or remove obvious decode waste.
