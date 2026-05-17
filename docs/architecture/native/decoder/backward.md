# Decoder Backward

Owner: `docs/architecture/native/decoder/backward.md`.
State: partial implementation, acceptance target.

## Acceptance Target

Accepted decoder training updates every trainable block tensor, not only token
embeddings and the LM head.

## Required Behavior

- Backward covers attention projections, output projection, MLP projections,
  RMSNorm weights, token embeddings, and LM head.
- Gradients are produced from CUDA-resident tape tensors and device loss
  gradients.
- Host reference code may run only as a parity oracle during tests or
  diagnostics, not as the gradient source for accepted optimizer steps.
- FP32 master weights and AdamW moments exist for every trainable tensor.
- BF16 shadows are refreshed from FP32 masters after AdamW updates.
- Checkpoint optimizer indexes include `master.NAME`, `adam_m.NAME`, and
  `adam_v.NAME` for every tensor.
- Resume restores counters, optimizer state, and tensor checksums.
- Acceptance tests must prove a non-embedding block weight changes on a tiny
  deterministic batch.

## Tape Ownership

Training forward stores the tensors needed by CUDA backward in
`DecoderCudaTape`, not in private layer scratch. The tape owns token/mask
device inputs, embeddings, per-layer residual outputs, final norm input,
final norm output, logits, grad logits, loss, and host capture buffers used for
report evidence. Layer-forward scratch remains an inference implementation
detail and is not the source of accepted optimizer gradients.

## Implementation Order

1. Introduce a decoder CUDA state that owns FP32 master weights, gradients,
   AdamW moments, BF16 shadows, activation buffers, and cuBLASLt workspace for
   decoder-shaped tensors.
2. Keep embedding and LM-head training on the proven dense substrate until the
   decoder state can produce matching loss and logits checks.
3. Add projection and MLP backward first because those tensors are already
   present in the forward substrate and are easiest to verify with deterministic
   tiny shapes.
4. Add attention, RMSNorm, and RoPE-adjacent backward after projection parity
   tests are stable.
5. Add checkpoint/resume coverage for every block tensor and optimizer slot.
6. Promote reports only after all trainable tensors have update evidence.

## Gradient Path

The backward pass runs in reverse decoder order:

1. CE grad logits into tied LM-head and final hidden gradients.
2. Final RMSNorm gradients into final norm weight and pre-norm hidden.
3. Per layer from last to first: final residual split, down projection,
   SwiGLU, gate/up projections, MLP RMSNorm, attention residual split,
   O projection, attention backward, inverse RoPE gradient, Q/K/V projection,
   and attention RMSNorm.
4. Embedding scatter-add into the tied token embedding registry tensor.

cuBLASLt owns projection gradients. Custom CUDA owns RMSNorm, RoPE, residual,
SwiGLU, embedding scatter, reductions, and BF16/FP32 conversion glue.

## Current Status

The current experimental path runs full decoder forward, CE loss, logits
capture, grad-logits, and diagnostic registry-gradient population on CUDA. It
fills FP32 registry gradients for the LM head or tied embedding table, token
embeddings, decoder block tensors, and final norm so registry CUDA AdamW can
exercise update plumbing.

Reports must identify this as `decoder_backward_backend=cuda_diagnostic_synthetic`
and `decoder_gradient_source=cuda_device_diagnostic`. They must keep
`accepted_cuda_training=false` until real chain-rule backward, cuDNN SDPA,
accepted decode evidence, the 40M RTX 3070 shape gate, logits/export/server
checks, and report promotion gates all pass.
