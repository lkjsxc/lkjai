# Dense P0 Runtime Contract

## Scope

This contract applies only to the accepted native dense CUDA trainer. The
transformer CUDA path remains experimental and must keep
`accepted_cuda_training=false`.

## Runtime Shape

Dense P0 uses:

- BF16 embedding and LM-head shadows with FP32 master weights.
- cuBLASLt forward, LM-head-gradient, and hidden-gradient GEMMs.
- A block-per-row FP32 softmax cross-entropy kernel with 256 threads per row.
- FP32 token scatter-add embedding gradients.
- One compute stream and one nonblocking copy stream.
- Three pinned batch slots for direct packed-cache reads and H2D staging.
- Deferred pinned loss readback materialized once per optimizer step.
- Single selected logits-row readback for checksum capture.

The dense batch row layout stays physical `batch x sequence`. Loss supervision
uses `loss_mask[pos + 1]`, predicts `tokens[pos + 1]`, divides by supervised
token count, and applies the caller-provided gradient scale.

## Report Fields

Schema stays stable. Accepted dense reports must include:

- `loss_kernel_backend="block_row_softmax_fp32"`
- `loss_readback_mode="optimizer_step_deferred_pinned"`
- `logits_readback_mode="single_row_capture"`
- `dense_logits_readback_bytes`
- `dense_stream_count=2`
- `dense_batch_slot_count=3`
- `copy_compute_overlap_enabled=true`
- `batch_staging_backend="triple_slot_pinned_direct_read"`

Existing dense fields remain required, including accepted CUDA status,
cuBLASLt/scatter backends, BF16 export/reference logits checks, deterministic
checksums, and unsupported decode behavior.

## Acceptance

A matched pre/post benchmark must use the same config digest, packed-cache
digest, batch size, sequence length, gradient accumulation, and CUDA
architecture flags. The post report is accepted when correctness gates pass and
throughput is not more than five percent worse than the baseline, with visible
improvement preferred in cross-entropy, backward, or H2D/compute timing.

## Later Roadmap

CUDA Graph capture, NCCL or ZeRO-style multi-GPU work, transformer CUDA
promotion, data-prefetch-only experiments, and native autoregressive decode are
outside this P0 runtime contract.
