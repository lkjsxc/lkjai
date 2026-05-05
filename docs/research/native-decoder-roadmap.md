# Native Decoder Research Synthesis

## Source

This file distills the latest ignored report,
`tmp/deep-research-report (28).md`, into durable canon. The ignored report is
not itself canonical because `tmp/` is local research scratch space.

## Conclusions

- The active product path is native C++/CUDA, not Python, Rust, or PyTorch.
- Dense BF16 CUDA training is an accepted foundation and benchmark substrate.
- The product target is the tied 40M `decoder` config on RTX 3070.
- Dense substrate evidence does not prove decoder readiness because the product
  model needs RoPE, RMSNorm, GQA, SwiGLU, tied embeddings, block backward, and
  KV-cache decode.
- The current decoder CUDA path is partial: embeddings and LM head train on the
  dense substrate, while the block forward substrate is probe-only.
- First decoder acceptance may use the correctness-first
  `cuda_causal_gqa_bf16_reference` attention backend.
- cuDNN SDPA is the preferred later attention performance backend after parity
  and timing are proven for the active GQA shape.
- TensorRT, TensorRT-LLM, CUTLASS, CUDA Graphs, and NCCL are profile or later
  scale work; none replaces the native decoder acceptance path.

## Implementation Order

1. Keep Compose verify green and strict under Docker.
2. Improve docs so every acceptance claim names the exact backend and non-goal.
3. Complete the decoder block forward substrate shape by shape.
4. Wire full decoder forward into training.
5. Add full decoder backward and FP32 AdamW state for every trainable tensor.
6. Add contiguous BF16 KV-cache decode without per-token device allocation.
7. Run the two-hour RTX 3070 decoder acceptance gate.

## Acceptance Defaults

- Hardware gate: RTX 3070 8GB, compute capability 8.6.
- Product config: `configs/native/decoder_40m_bf16_3070.json`.
- Training config: `configs/training/decoder_2h_40m_3070.json`.
- Accepted report fields must include `implementation_status=accepted`,
  `accepted_cuda_training=true`, `decoder_cuda_slice=full_decoder`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and `decode_backend=cuda_kv_cache`.

## Non-Claims

- Partial decoder reports are not accepted training evidence.
- Host-reference recompute decode is not accepted serving evidence.
- Blackwell or RTX 5090 runs are profile evidence until the RTX 3070 gate also
  passes.
