# Current State

Owner: `docs/current-state.md`.
State: canonical orientation map.

## Accepted Foundation

Dense BF16 CUDA training is the accepted native substrate. It covers
embedding and LM-head training, FP32 master weights and AdamW moments, BF16
device shadows, packed-cache input, checkpoint/export, logits checks, and
benchmark continuity. Dense evidence does not prove decoder block training or
chat-quality serving.

Product training, serving, runtime, verification, and benchmark paths remain
native C++/CUDA. Python is limited to corpus acquisition and other non-product
preparation work.

## Decoder Status

The `decoder` model kind is the product target, but the current CUDA training
slice is partial:

- `implementation_status=partial_cuda`
- `decoder_cuda_slice=embedding_lm_head`
- `decoder_backward_backend=not_implemented`
- `kv_cache_backend=none`

The slice trains embeddings and the LM head through the dense CUDA substrate.
The decoder block path is forward-only evidence: RMSNorm, RoPE, Q/K/V/O
projections, causal GQA attention, residuals, SwiGLU, and down projection have
CUDA parity coverage, but their weights are not trained by the current slice.

Host-reference recompute decode is partial usability only. It may produce
decoder `choices`, but it is not accepted CUDA KV-cache serving evidence.

## Quality Limits

Current reports must not claim accepted decoder CUDA training. LM-head-only
updates are insufficient for decoder acceptance. Accepted decoder reports must
prove real block-weight updates, full block backward, FP32 optimizer coverage
for every trainable decoder tensor, export/logits/server checks, and native
KV-cache decode.

Large-model profiles remain planning and benchmark evidence until the local
decoder acceptance lane passes. A larger GPU result cannot relax the RTX 3070
gate by itself.

## Next Target

The next product acceptance target is the tied 40M decoder on RTX 3070:

- native config: `configs/native/decoder_40m_bf16_3070.json`
- training config: `configs/training/decoder_2h_40m_3070.json`
- required report fields include `implementation_status=accepted`,
  `accepted_cuda_training=true`, `decoder_cuda_slice=full_decoder`,
  `decoder_block_weight_changed=true`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and
  `decode_backend=cuda_kv_cache`

The latest deep research report under `tmp/deep-research-report (42).md`
supports this order: keep the dense substrate accepted, tighten partial
decoder reporting, finish the 40M RTX 3070 decoder gate first, then add
1.5B-3B and larger profile evidence.
