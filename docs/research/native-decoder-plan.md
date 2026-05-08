# Native Decoder Plan Canon

## Source

This file distills ignored reports `tmp/deep-research-report (31).md` and the
decoder-relevant conclusions from `tmp/deep-research-report (28).md` into
durable canon. Reports under `tmp/` are source evidence, not canonical docs.

## Canon Decision

- The active train/serve product path is native C++/CUDA.
- Dense BF16 CUDA training is the accepted foundation and regression substrate.
- The first accepted same-model product target is
  `configs/native/decoder_40m_bf16_3070.json` with
  `configs/training/decoder_2h_40m_3070.json`.
- The target hardware gate is RTX 3070 8GB, compute capability 8.6.
- Higher-memory or newer-GPU runs are profile evidence until the RTX 3070 gate
  also passes.

## Foundation Versus Product

- Dense accepted evidence covers embeddings, LM head, packed-cache training,
  BF16 CUDA math, FP32 AdamW state, checkpoint/export, logits checks, and native
  unsupported-decode behavior.
- Dense substrate evidence does not prove decoder readiness because the product
  model needs RoPE, RMSNorm, GQA, SwiGLU, tied embeddings, block backward, and
  KV-cache decode.
- The current decoder CUDA path is partial: embeddings and LM head train on the
  dense substrate, while the block forward substrate is probe-only.
- Host-reference recompute decode is partial serving evidence only.

## Backend Ownership

- cuBLASLt owns QKV, output, FFN, and LM-head GEMMs unless profiling proves a
  narrower replacement.
- Custom CUDA kernels own RMSNorm, RoPE, residual paths, SwiGLU glue, loss,
  BF16/FP32 casts, AdamW helpers, KV-cache writes, filtering, and sampling.
- First full decoder acceptance may use
  `attention_backend=cuda_causal_gqa_bf16_reference`, the correctness-first CUDA
  causal GQA path.
- cuDNN SDPA is the preferred performance attention backend after parity and
  timing are proven for the active GQA shape.
- CUTLASS, CUDA Graphs, TensorRT, TensorRT-LLM, and NCCL are profile, serving,
  or later scale work; none replaces first native decoder acceptance.

## Implementation Order

1. Keep Compose verify green and strict under Docker.
2. Keep report fields explicit so partial CUDA cannot look accepted.
3. Complete the decoder block forward substrate and wire it into training.
4. Add full decoder backward and FP32 AdamW state for every trainable tensor.
5. Add tied-embedding optimizer/export alias handling for the product config.
6. Add contiguous BF16 KV-cache decode without per-token device allocation.
7. Add streaming output after accepted native decode exists.
8. Run the two-hour RTX 3070 decoder acceptance gate.

## Acceptance Defaults

- Hardware gate: RTX 3070 8GB, compute capability 8.6.
- Product config: `configs/native/decoder_40m_bf16_3070.json`.
- Training config: `configs/training/decoder_2h_40m_3070.json`.
- Product shape: vocab `8192`, context `1024`, layers `10`, hidden `576`,
  heads `8`, KV heads `2`, head dim `72`, FFN `1536`, SwiGLU, RoPE,
  RMSNorm, BF16, tied embeddings.
- Accepted report fields must include `implementation_status=accepted`,
  `accepted_cuda_training=true`, `decoder_cuda_slice=full_decoder`,
  `decoder_backward_backend=cuda_full_decoder`,
  `kv_cache_backend=cuda_contiguous_bf16`, and `decode_backend=cuda_kv_cache`.
- Accepted evidence must prove finite loss, nonzero non-embedding weight
  changes, checkpoint/resume/export/logits checks, native server chat `choices`,
  exact command/config paths, git commit, GPU, driver, CUDA, cuDNN, attention
  backend, GEMM backend, workspace sizes, and timing breakdowns.

## Non-Claims

- Dense 20M/40M configs are foundation and profiling targets, not same-model
  decoder acceptance targets.
- Partial decoder reports are not accepted full-decoder training evidence.
- Host-reference recompute decode is not accepted serving evidence.
- Blackwell or RTX 5090 runs are profile evidence until the RTX 3070 gate also
  passes.
- Quantized or TensorRT-family inference does not satisfy native BF16 training
  or native KV-cache decode acceptance.

## Open Questions

- Larger 100M-150M decoder profiles remain recommendations, not audited current
  repo contracts.
- Public deployment hardening remains out of scope until native BF16 decode is
  accepted.
