# Dense BF16 Optimization Contract

## Scope

This contract covers the accepted dense native CUDA trainer only. It does not
promote transformer CUDA training, chat decode, graph capture, multi-GPU
collectives, or alternate precision modes.

## Current Baseline

The accepted CUDA path is dense embedding plus LM head training. It uses BF16
shadow weights for embedding gather and LM-head logits, FP32 master weights and
gradients, cuBLASLt for the forward logits GEMM, custom CUDA loss and optimizer
kernels, packed-cache v2 input, artifact v2 output, and train-report schema v3.

Transformer reports remain experimental with
`accepted_cuda_training=false`; dense reports remain accepted with
`accepted_cuda_training=true`.

## Accepted Runtime

The accepted dense path uses:

- cuBLASLt LM-head gradient GEMM:
  `grad_head += grad_logits^T * hidden`.
- cuBLASLt hidden-gradient GEMM:
  `d_hidden = grad_logits * lm_head`.
- Token scatter-add embedding gradient kernel:
  `atomicAdd(grad_emb[token, hidden], d_hidden[row, hidden])`.
- Block-per-row FP32 softmax cross entropy.
- Deferred pinned loss readback at optimizer-step boundaries.
- Single-row logits capture.
- Two CUDA streams with three pinned packed-cache batch slots.

The loss kernel still applies `grad_scale`. Head-gradient GEMM uses
`beta=1`, and scatter-add accumulates into FP32 embedding gradients so
gradient accumulation semantics match the previous dense trainer contract.

## Non-Goals

- Direct transformer CUDA forward or backward promotion.
- Autoregressive decode or KV cache.
- CUDA Graph capture or timing modes beyond existing CUDA events.
- NCCL, ZeRO, or other multi-GPU behavior.
- FP16 fallback or precision-mode selection beyond the documented BF16 dense
  path.

## Report Contract

Schema v3 remains current. Dense reports add:

- `backward_backend="cuda_bf16_cublaslt_scatter"`
- `backward_gemm_enabled=true`
- `embedding_grad_backend="token_scatter_add_fp32"`
- `dense_step_logits_bytes`
- `dense_step_grad_logits_bytes`
- `dense_step_d_hidden_bytes`
- `cublaslt_workspace_bytes`
- `loss_kernel_backend="block_row_softmax_fp32"`
- `loss_readback_mode="optimizer_step_deferred_pinned"`
- `logits_readback_mode="single_row_capture"`
- `dense_logits_readback_bytes`
- `dense_stream_count=2`
- `dense_batch_slot_count=3`
- `copy_compute_overlap_enabled=true`
- `batch_staging_backend="triple_slot_pinned_direct_read"`

Consumers must treat those fields as additive schema-v3 fields. A schema v4 is
not required unless a future change removes or redefines existing fields.

## Acceptance

- `docker compose --progress quiet --profile verify run --rm verify` passes on
  the RTX 3070 acceptance profile.
- Dense train reports remain accepted CUDA training.
- Transformer train reports remain experimental.
- `/v1/chat/completions` returns HTTP `422` with no `choices` for dense and
  transformer artifacts until decode lands.
- Dense logits/reference checks, resume equivalence, and exported checksum
  determinism pass.
- A matched pre/post benchmark records the same config digest, batch size,
  sequence length, gradient accumulation, cache, and CUDA architecture flags.
  The slice is accepted when backward time improves or total throughput is not
  more than 5 percent worse.

If cuBLASLt accumulation order creates small dense parity drift, dense
gradient/update tolerances may be relaxed only up to `5e-3` and the reason must
be documented here. Larger drift blocks the change.

## Roadmap

Later phases are, in order: explicit precision modes, transformer CUDA forward
kernels, transformer backward, accepted transformer CUDA reports, native decode
with KV cache, CUDA Graph buckets, and NCCL/ZeRO-style work after single-GPU
transformer gates pass.

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
