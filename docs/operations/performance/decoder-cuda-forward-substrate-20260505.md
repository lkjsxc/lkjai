# Decoder CUDA Forward Substrate 2026-05-05

## Baseline

Baseline commits referenced by this substrate batch:

- `9f23419`
- `33f0df2`
- `44498cb`
- `3fefbee`
- `d4319d2`

New commit: `TBD after commit`.

## Changed Primitives

This batch adds a decoder block forward-substrate probe, not full decoder
training. The probe validates decoder shape metadata and runs deterministic
small tensors through:

- BF16 RMSNorm with FP32 reduction.
- BF16 RoPE for Q and K layouts `[B,S,heads,D]`.
- row-major BF16 cuBLASLt projections for Q, K, V, and O.
- BF16 correctness-first causal GQA attention.
- BF16 SwiGLU glue for `silu(gate) * up`.

Measured checks are finite BF16 outputs, CPU-comparable RMSNorm and RoPE CTest
coverage, Q/K/V/O metadata validation, and serialized report-field coverage.

## Non-Claims

This is not accepted full decoder CUDA training. It does not provide:

- cuDNN SDPA performance attention,
- decoder block backward,
- full optimizer coverage for block tensors,
- KV-cache decode,
- 40M decoder acceptance.

## Report Fields

Decoder reports remain `accepted_cuda_training=false` and emit:

- `decoder_block_backend=cuda_forward_partial`
- `rmsnorm_backend=cuda_bf16_fp32_reduce`
- `rope_backend=cuda_bf16`
- `qkv_projection_backend=cuda_bf16_cublaslt`
- `attention_backend=cuda_causal_gqa_bf16_reference`
- `mlp_backend=cuda_swiglu_partial`
- `decoder_backward_backend=not_implemented`
- `kv_cache_backend=none`
- `decode_backend=host_reference_recompute`

## Verification

Required verification command:

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```

The deterministic CTest target is
`native_decoder_cuda_block_forward_substrate`.

## Next Bottlenecks

- cuDNN SDPA for faster causal/GQA attention.
- Full decoder block backward.
- Optimizer state for block tensors.
- Contiguous BF16 KV-cache decode.
