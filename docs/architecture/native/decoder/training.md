# Decoder Training

Owner: `docs/architecture/native/decoder/training.md`.
State: acceptance contract.

## Goal

Train a same-model decoder through native C++/CUDA without Python or PyTorch in
the product path. Accepted chat comes after real CUDA KV-cache decode evidence.

## CUDA Ownership

Accepted full decoder CUDA training requires:

- cuBLASLt owns QKV, output, MLP, and LM-head GEMMs.
- First acceptance may use `attention_backend=cuda_causal_gqa_bf16_reference`,
  a correctness-first CUDA causal GQA path with FP32 score and softmax
  accumulation.
- cuDNN SDPA remains the later performance backend when frontend integration and
  parity are complete for the active shape.
- Custom kernels own RMSNorm/residual fusion, RoPE, SwiGLU glue, CE loss,
  BF16/FP32 casts, AdamW helpers, KV writes, logits filtering, and sampling.
- FP32 master weights and Adam moments are the optimizer state.
- BF16 shadows are refreshed after optimizer updates and exported for serving.

Accepted decoder training covers embeddings, tied LM head, every decoder block
tensor, and final norm. Reports must prove positive non-embedding and
decoder-block deltas before using accepted backend names.

The decoder path owns registry tensors, BF16 shadows, AdamW moments, and
diagnostic CUDA buffers for every trainable decoder tensor. Smoke reports prove
the contract shape; accepted evidence still requires the documented RTX 3070
run and route checks.
The implementation must prefer correctness evidence over kernel cleverness:
cuBLASLt remains the GEMM owner, while custom CUDA covers pointwise kernels,
attention glue, loss, optimizer helpers, sampling, and KV-cache operations.

## Public Invocation

```bash
lkjai-native-train --train --mode decoder \
  --config configs/native/decoder_40m_bf16_3070.json \
  --tokenizer data/train/tokenizer/tokenizer.json \
  --packed-cache data/train/datasets/packed/train-causal_lm_full-seq1024
```

Environment equivalents are `TRAIN_MODEL_KIND=decoder`,
`TRAIN_NATIVE_CONFIG`, `TRAIN_TOKENIZER`, `TRAIN_PACKED_CACHE_DIR`, and
`TRAIN_TARGET_SECONDS`.

## Wall-Clock Stop

Decoder training supports a native deadline:

- CLI: `--target-seconds N`
- Environment: `TRAIN_TARGET_SECONDS=N`
- Training config: `target_seconds`

The trainer checks the deadline before each optimizer step. When the deadline is
hit it writes `latest`, `final`, export, served artifact, and a report with
`stop_reason=wall_clock_deadline`, `deadline_hit=true`, and `target_seconds`.

## Report Fields

Decoder reports use stable schema with additive fields:

- `model_kind=decoder`
- `implementation_status`
- `accepted_cuda_training`
- `decoder_cuda_path`
- `decoder_cuda_slice`
- `decoder_block_backend`
- `decoder_block_forward_in_training`
- `decoder_block_forward_steps`
- `decoder_forward_probe`
- `embedding_weight_changed`
- `lm_head_weight_changed`
- `decoder_block_weight_changed`
- `non_embedding_weight_changed`
- `rmsnorm_backend`
- `rope_backend`
- `qkv_projection_backend`
- `forward_backend`
- `backward_backend`
- `attention_backend`
- `mlp_backend`
- `decoder_backward_backend`
- `matmul_backend`
- `optimizer_backend`
- `decode_supported`
- `target_seconds`
- `deadline_hit`
- `stop_reason`
- `kv_cache_backend`
- `decode_backend`
- `kv_cache_prefill_allocated_bytes`
- `kv_cache_steady_state_token_allocations`
- `embedding_tying`
- `trainable_tensor_count`

Reports are accepted only when `accepted_cuda_training=true`,
`implementation_status=accepted`, `decoder_cuda_slice=full_decoder`, CUDA
forward/backward/attention backends are present, `decode_supported=true`,
`logits_check_passed=true`, finite loss, `steps > 0`, `loss_tokens > 0`,
`trainable_weight_changed=true`, nonzero non-embedding block/final-norm weight
change, `decoder_block_weight_changed=true`, checkpoint/export/logits/server
checks pass, tied embedding alias metadata is present, accepted KV-cache and
decode backend names are reported, KV prefill allocation is positive, steady
state per-token device allocation is zero, and the documented benchmark gate
passes. LM-head-only updates do not satisfy decoder block-training acceptance.

No report may emit `implementation_status=accepted`,
`decoder_cuda_slice=full_decoder`,
`decoder_backward_backend=cuda_full_decoder`,
`decode_backend=cuda_kv_cache`, or
`kv_cache_backend=cuda_contiguous_bf16` unless block backward, optimizer
coverage for all trainable decoder tensors, accepted logits/export/server
checks, and CUDA KV-cache decode execute. Sidecars such as
`decoder_acceptance.json` are written only for the accepted 40M RTX 3070
training configuration after report fields pass.

The two-hour RTX 3070 run is the acceptance lane. Code must not promote a
report solely because `target_seconds > 0`.

## Current Status

The decoder lane is promoted only through this document's acceptance contract.
Historical partial reports remain useful regression evidence. Any future
partial report must keep `accepted_cuda_training=false`, avoid accepted backend
names, and avoid accepted decode backend names.

The report contract rejects partial slices, missing logits evidence, missing
served artifacts, missing block-weight deltas, untied product configs, and KV
decode without allocation accounting.
