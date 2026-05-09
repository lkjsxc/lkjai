# Decoder Training

Owner: `docs/architecture/native/decoder/training.md`.
State: future acceptance contract.

## Goal

Train a same-model chat-capable decoder through native C++/CUDA without Python
or PyTorch in the product path.

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

The current partial decoder slice trains only embeddings and the LM head. A
decoder forward-substrate probe now runs first and covers RMSNorm, RoPE,
Q/K/V/O projections, causal GQA attention, attention residual, MLP RMSNorm,
SwiGLU, down projection, and final residual on deterministic tensors. It must
not be described as accepted full decoder CUDA training because block backward,
block optimizer state, block-weight updates, and KV-cache decode are still
absent.

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
- `decoder_block_weight_changed`
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
- `embedding_tying`
- `trainable_tensor_count`

Reports are accepted only when `accepted_cuda_training=true`,
`implementation_status=accepted`, `decoder_cuda_slice=full_decoder`, CUDA
forward/backward/attention backends are present, finite loss and nonzero
non-embedding weight change are proven, `decoder_block_weight_changed=true`,
checkpoint/export/logits/server checks pass, tied embedding alias metadata is
present, and the documented benchmark gate passes. LM-head-only updates do not
satisfy decoder block-training acceptance.

## Current Status

Foundation server-contract work reports `implementation_status=experimental`,
`decoder_cuda_path=false`, `attention_backend=host_reference`, and validates
decoder config/artifact/logits/server `choices` contracts only.

Commit `01dac62` adds the first partial CUDA-backed decoder training slice:
`implementation_status=partial_cuda`, `decoder_cuda_path=true`,
`decoder_cuda_slice=embedding_lm_head`, `matmul_backend=cublaslt`, and
`optimizer_backend=cuda_adamw_fp32`. It copies and checksums the real
byte-level BPE tokenizer into decoder artifacts, sets `decode_supported=true`,
and serves decoder chat through native prompt serialization and tokenization.

The current forward-substrate batch keeps acceptance unchanged while reporting
`decoder_block_backend=cuda_forward_partial`,
`rmsnorm_backend=cuda_bf16_fp32_reduce`, `rope_backend=cuda_bf16`,
`qkv_projection_backend=cuda_bf16_cublaslt`,
`mlp_backend=cuda_swiglu_partial`, and
`decoder_backward_backend=not_implemented`.

The slice now also executes one real decoder block forward on the first training
batch and reports `decoder_block_forward_in_training=true` with the executed
step count. Training reports remain partial until full forward, backward,
optimizer coverage, block-weight updates, and KV-cache decode are wired into
the trainer. Foundation server contract and embedding/head CUDA training are
not accepted full decoder training.
