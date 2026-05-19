# Decoder Training

Owner: `docs/architecture/native/decoder/training.md`.
State: acceptance contract; field matrix in [acceptance.md](acceptance.md).

## Goal

Train a same-model decoder through native C++/CUDA without Python or PyTorch in
the product path. Accepted chat comes after real CUDA KV-cache decode evidence.

## CUDA Ownership

Accepted full decoder CUDA training requires:

- cuBLASLt owns QKV, output, MLP, and LM-head GEMMs.
- Accepted reports use `attention_backend=cudnn_sdpa_bf16_gqa` for BF16
  causal grouped-query attention.
- `cuda_causal_gqa_bf16_reference` remains a diagnostic fallback and parity
  oracle, not an accepted attention backend.
- Custom kernels own RMSNorm/residual fusion, RoPE, SwiGLU glue, CE loss,
  BF16/FP32 casts, AdamW helpers, KV writes, logits filtering, and sampling.
- FP32 master weights and Adam moments are the optimizer state.
- BF16 shadows are refreshed after optimizer updates and exported for serving.

Accepted decoder training covers embeddings, tied LM head, every decoder block
tensor, and final norm. Reports must prove positive non-embedding and
decoder-block deltas before using accepted backend names.

The decoder path owns registry tensors, BF16 shadows, AdamW moments, tape
buffers, and diagnostic CUDA buffers for every trainable decoder tensor. Smoke
reports prove the contract shape; accepted evidence still requires the
documented RTX 3070 run and route checks.

Training tape population is implemented for chain-rule block backward. The
forward pass persists the per-layer tensors that backward consumes:
`attn_norm_input` for the pre-attention RMSNorm input, `attn_norm` for its
output, `q_rope` and `k_rope` after RoPE, `mlp_norm_input` for the pre-MLP
RMSNorm input, and `mlp_norm` for its output.

The current experimental training slice uses CUDA for the full decoder forward
stack, FP32 logits, CE loss, grad-logits, supervised-row logit capture,
chain-rule backward, and registry-wide CUDA AdamW over device FP32 masters,
AdamW moments, gradients, and BF16 shadows. Its truthful report fields
are `implementation_status=experimental`, `accepted_cuda_training=false`,
`decoder_cuda_slice=full_decoder`, `forward_backend=cuda_full_decoder`,
`backward_backend=cuda_decoder_chain_rule`,
`optimizer_backend=cuda_adamw_fp32_registry`,
`decoder_backward_backend=cuda_decoder_chain_rule`,
`decoder_gradient_source=cuda_device`, and
`attention_backend=cuda_causal_gqa_bf16_reference`.
The implementation must prefer correctness evidence over kernel cleverness:
cuBLASLt remains the GEMM owner, while custom CUDA covers pointwise kernels,
attention glue, loss, optimizer helpers, sampling, and KV-cache operations.

## Device-Origin Gradient Rule

Accepted decoder gradients must originate from the CUDA backward path that
consumes the recorded decoder tape. Copying block gradients from
`transformer_backward(...)`, host parity helpers, or diagnostic probes is not
accepted evidence even when CUDA AdamW updates the registry afterward.

The CUDA backward path must populate FP32 gradient buffers for:

- tied token embedding and LM-head rows,
- final RMSNorm,
- every layer RMSNorm,
- Q, K, V, and O projections,
- gate, up, and down MLP projections.

Reports may claim `decoder_backward_backend=cuda_full_decoder` only after the
training step uses chain-rule device gradients from the recorded decoder tape
for optimizer input. Diagnostic CUDA helper gradients must keep
`decoder_backward_backend=cuda_diagnostic_synthetic` and
`decoder_gradient_source=cuda_device_diagnostic`.

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
- `decoder_gradient_source`
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

Reports are accepted only when [acceptance.md](acceptance.md) passes:
`accepted_cuda_training=true`,
`implementation_status=accepted`, `decoder_cuda_slice=full_decoder`, CUDA
forward/backward backends are present, `attention_backend=cudnn_sdpa_bf16_gqa`,
`decoder_gradient_source=cuda_device`,
`decode_supported=true`,
`logits_check_passed=true`, finite loss, `steps > 0`, `loss_tokens > 0`,
`trainable_weight_changed=true`, nonzero non-embedding block/final-norm weight
change, `decoder_block_weight_changed=true`, checkpoint/export/logits/server
checks pass, tied embedding alias metadata is present, accepted KV-cache and
decode backend names are reported, KV prefill allocation is positive, steady
state per-token device allocation is zero, and the documented benchmark gate
passes. LM-head-only updates do not satisfy decoder block-training acceptance.

No report may emit `implementation_status=accepted`,
`attention_backend=cudnn_sdpa_bf16_gqa`, `decode_backend=cuda_kv_cache`, or
`kv_cache_backend=cuda_contiguous_bf16` unless block backward, optimizer
coverage for all trainable decoder tensors, accepted logits/export/server
checks, and CUDA KV-cache decode execute. Sidecars such as
`decoder_acceptance.json` are written only for the accepted 40M RTX 3070
training configuration after report fields pass.

The two-hour RTX 3070 run is the acceptance lane. Code must not promote a
report solely because `target_seconds > 0`.

## Current Status

The decoder lane is promoted only through the acceptance matrix and this
document's training contract.
Historical partial reports remain useful regression evidence. Any future
partial report must keep `accepted_cuda_training=false`, avoid accepted
attention names, and avoid accepted decode backend names.

The report contract rejects partial slices, missing logits evidence, missing
served artifacts, missing block-weight deltas, untied product configs, and KV
decode without allocation accounting.
