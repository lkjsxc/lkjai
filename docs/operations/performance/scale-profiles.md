# Scale Profiles

Owner: `docs/operations/performance/scale-profiles.md`.
State: profile evidence contract.

## Purpose

Keep large-model planning explicit without weakening the RTX 3070 acceptance
gate.

## Local Acceptance Lane

- First accepted product target: 40M decoder on RTX 3070 8GB.
- Required path: native BF16 training, full decoder backward, export, logits
  check, native server chat, and contiguous BF16 KV-cache decode.
- A larger GPU run is profile evidence until the same contract passes on RTX
  3070 or a new acceptance lane is documented and verified.

## Profile Lane

| Profile | Role | Entry condition | Evidence required |
|---|---|---|---|
| 1.5B-3B dense decoder | First practical large profile | Accepted 40M decoder | Train and serve reports on one 80GB-class GPU or equivalent |
| 7B dense decoder | Multi-GPU profile | Tensor-parallel and checkpointing contracts | Scaling, loss, memory, and decode evidence on 2-4 linked GPUs |
| 14B-20B dense decoder | Upper profile | Verified 7B evidence | Distributed training report, communication timing, and cost notes |

Profiles guide design and benchmarking. They do not become accepted defaults by
size alone.

## Research Synthesis

The latest deep research pass reinforces the existing order: finish the local
decoder contract before chasing larger profile sizes. The report's useful
distinction is operational:

- `40M` on RTX 3070 is the acceptance lane because it keeps correctness,
  reports, export, and serving reproducible on the local gate.
- `1.5B-3B` is the first practical large profile after acceptance because it
  can fit a single 80GB-class card with careful activation and optimizer memory
  discipline.
- `7B` requires tensor parallelism or linked multi-GPU training before it is a
  responsible training target.
- `14B-20B` is profile-only until distributed execution, checkpointing, and
  communication timing are already stable at smaller scale.

Do not let a successful larger run weaken the local gate. Larger profile
evidence adds confidence only when the same report fields and API contracts are
preserved.

## Memory Rule

Use this planning estimate for native BF16 training with FP32 optimizer state:

- BF16 forward weights: about `2` bytes per parameter.
- FP32 master weights: about `4` bytes per parameter.
- FP32 gradients: about `4` bytes per parameter.
- FP32 AdamW moments: about `8` bytes per parameter.
- Static training state floor: about `18` bytes per parameter, before
  activations, workspaces, fragmentation, and communication.

BF16 inference weights are about `2` bytes per parameter plus KV cache.

## Hardware Tiers

- RTX 3070: correctness and acceptance gate.
- RTX 4090 or RTX 5090: profile targets for headroom and kernel behavior.
- A100 80GB or H100-class cards: practical 1.5B-3B profile hosts.
- 2-4 linked 80GB GPUs: required planning tier for 7B training.
- Larger linked systems: profile-only tier for 14B-20B work.

## Ordering

1. Finish accepted single-GPU 40M decoder.
2. Add memory-accounted large-profile configs and benchmark output fields.
3. Add dataset lineage reports for every large-profile packed cache.
4. Add tensor parallelism.
5. Add activation checkpointing.
6. Add pipeline staging.
7. Add communication overlap.
8. Add optimizer sharding after the prior distributed contracts are measurable.
9. Evaluate TensorRT-family inference only after native KV-cache decode works.

Large-profile evidence must keep `profile-only` status in the narrative until
the active acceptance contract linked from
[../../architecture/native/decoder/training.md](../../architecture/native/decoder/training.md)
passes for the local lane.
