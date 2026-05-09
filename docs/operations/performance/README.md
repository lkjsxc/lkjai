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
- [benchmark-suite.md](benchmark-suite.md): substrate, train-step, bounded
  training, and serving measurement layers.
- [benchmark-output.md](benchmark-output.md): JSON and CSV output shape for
  native benchmark tools.
- [evidence.md](evidence.md): required fields for accepted performance records.
- [dense-evidence-record.md](dense-evidence-record.md): curated benchmark
  evidence format for dense CUDA throughput changes.
- [dense-bf16-cuda-training-report.md](dense-bf16-cuda-training-report.md):
  two-hour dense BF16 CUDA workflow, commands, and current cache blocker.
- [dense-substrate-20260504.md](dense-substrate-20260504.md): RTX 3070
  evidence for dense autotune, async allocation, deferred timing, and cache
  reporting.
- [decoder-cuda-forward-substrate-20260505.md](decoder-cuda-forward-substrate-20260505.md):
  decoder forward-only CUDA primitive substrate evidence and non-claims.
- [hardware-profiles.md](hardware-profiles.md): RTX 3070 acceptance gate and
  RTX 5090 benchmark profile.
- [scale-profiles.md](scale-profiles.md): 1.5B-3B, 7B, and 14B-20B profile
  planning without weakening the 3070 gate.
- [profiling.md](profiling.md): Nsight and NVTX protocol for native work.
- [kernel-plan.md](kernel-plan.md): vendor-library and native CUDA
  escalation order.
- [validation.md](validation.md): numerical, resume, and server acceptance
  matrix for CUDA work.

## Active Priority

Training speed is the first priority. Inference improvements are accepted when
they share the same model/cache foundations or remove obvious decode waste.
