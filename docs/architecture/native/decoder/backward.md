# Decoder Backward

## Acceptance Target

Accepted decoder training updates every trainable block tensor, not only token
embeddings and the LM head.

## Required Behavior

- Backward covers attention projections, output projection, MLP projections,
  RMSNorm weights, token embeddings, and LM head.
- FP32 master weights and AdamW moments exist for every trainable tensor.
- BF16 shadows are refreshed from FP32 masters after AdamW updates.
- Checkpoint optimizer indexes include `master.NAME`, `adam_m.NAME`, and
  `adam_v.NAME` for every tensor.
- Resume restores counters, optimizer state, and tensor checksums.
- Acceptance tests must prove a non-embedding block weight changes on a tiny
  deterministic batch.

## Current Status

The partial CUDA slice trains embeddings and the LM head only. Reports must keep
`decoder_backward_backend=not_implemented` until full block backward and
optimizer coverage are implemented and verified.
