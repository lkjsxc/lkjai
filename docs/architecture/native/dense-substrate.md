# Dense CUDA Substrate

## Purpose

This is the accepted dense BF16 CUDA optimization surface. It is the shared
substrate that future transformer work may reuse, but this file does not
promote transformer CUDA, decode, CUDA Graphs, or NCCL.

## Current Hot Path

One dense microstep is:

1. Copy one packed-cache batch from pinned host memory to device memory.
2. Gather BF16 token embeddings into `[rows, hidden]`.
3. Compute FP32 logits with cuBLASLt BF16 GEMM.
4. Run FP32 row-wise softmax cross entropy and write FP32 grad logits.
5. Convert BF16 hidden and LM head shadows to FP32 operands.
6. Run cuBLASLt head-gradient and hidden-gradient GEMMs.
7. Scatter-add FP32 hidden gradients into token embedding gradients.
8. Apply AdamW from FP32 master weights and moments, then refresh BF16 shadows.

`rows = batch_size * seq_len`. The largest transient buffers scale with
`rows * vocab_size`, not only with parameter count.

## Optimization Contract

- Keep BF16 shadows for GEMM inputs and exports.
- Keep FP32 master weights, gradients, AdamW moments, loss, softmax stats, and
  checksums.
- Keep dense train reports accepted only when `model_kind=dense` and
  `accepted_cuda_training=true`.
- Keep transformer reports experimental until real device-resident transformer
  forward and backward kernels pass their own gates.
- Keep `/v1/chat/completions` unsupported with HTTP `422` and no `choices`
  until native decode lands.

## Runtime Tunables

Dense tuning is controlled by environment variables so benchmark scripts can
sweep modes without changing artifact or HTTP contracts:

- `LKJAI_DENSE_AUTOTUNE=heuristic|benchmark|off`
- `LKJAI_DENSE_WORKSPACE_SWEEP=BYTES[,BYTES...]`
- `LKJAI_DENSE_ALLOCATOR=auto|async|legacy`
- `LKJAI_DENSE_TIMING=deferred|legacy`

Defaults must be safe on RTX 3070. Unsupported or invalid values fall back to
the documented default and must be visible in additive report fields.

## Immediate Priorities

1. Cache cuBLASLt choices by device, shape, dtype, transpose, order, and
   workspace budget.
2. Expose allocator backend, async allocation support, workspace high-water
   bytes, and workspace reallocations.
3. Prefer deferred CUDA-event timing over phase-local synchronizations.
4. Reuse the FP32 LM-head conversion across microsteps inside one optimizer
   step, invalidating it after AdamW refresh.
5. Keep chunked vocab CE and token-bucketed scatter as later measured kernels
   behind explicit report fields when implemented.

## Acceptance

- Docker Compose verify passes.
- Dense learning-control remains promotable on RTX 3070.
- Bounded 40M compatibility start check succeeds and remains
  `run_purpose=bounded_compatibility_start_check`.
- Dense logits reference, resume equivalence, export checksums, and unsupported
  server decode behavior remain unchanged.
