# Decoder Training

## Goal

Train a same-model chat-capable decoder through native C++/CUDA without Python
or PyTorch in the product path.

## CUDA Ownership

- cuBLASLt owns QKV, output, MLP, and LM-head GEMMs.
- cuDNN SDPA owns BF16 causal/GQA attention when capability checks pass.
- Custom kernels own RMSNorm/residual fusion, RoPE, SwiGLU glue, CE loss,
  BF16/FP32 casts, AdamW helpers, KV writes, logits filtering, and sampling.
- FP32 master weights and Adam moments are the optimizer state.
- BF16 shadows are refreshed after optimizer updates and exported for serving.

## Wall-Clock Stop

Decoder training supports a native deadline:

- CLI: `--target-seconds N`
- Environment: `TRAIN_TARGET_SECONDS=N`
- Training config: `target_seconds`

The trainer checks the deadline before each optimizer step. When the deadline is
hit it writes `latest`, `final`, export, served artifact, and a report with
`stop_reason=wall_clock_deadline`, `deadline_hit=true`, and `target_seconds`.

## Report Fields

Decoder reports use schema version `3` with additive fields:

- `model_kind=decoder`
- `implementation_status`
- `accepted_cuda_training`
- `decoder_cuda_path`
- `decoder_cuda_slice`
- `decoder_block_backend`
- `forward_backend`
- `backward_backend`
- `attention_backend`
- `matmul_backend`
- `optimizer_backend`
- `decode_supported`
- `target_seconds`
- `deadline_hit`
- `stop_reason`
- `kv_cache_backend`

Reports are accepted only when `accepted_cuda_training=true`,
`implementation_status=accepted`, `decoder_cuda_slice=full_decoder`, CUDA
forward/backward/attention backends are present, finite loss and nonzero weight
change are proven, checkpoint/export/logits/server checks pass, and the
documented benchmark gate passes.

## Current Status

P0 server-contract work reports `implementation_status=experimental`,
`decoder_cuda_path=false`, `attention_backend=host_reference`, and validates
decoder config/artifact/logits/server `choices` contracts only.

Commit `01dac62` adds the first partial CUDA-backed decoder training slice:
`implementation_status=partial_cuda`, `decoder_cuda_path=true`,
`decoder_cuda_slice=embedding_lm_head`, `matmul_backend=cublaslt`, and
`optimizer_backend=cuda_adamw_fp32`. Token embeddings and LM head are trained
with device-resident BF16 shadows, FP32 masters, FP32 Adam state, and reusable
CUDA workspaces.

The decoder blocks remain `decoder_block_backend=static_reference`, attention
remains `attention_backend=not_implemented`, and `accepted_cuda_training=false`.
P0 server contract is not the accepted CUDA decoder trainer, and partial CUDA
embedding/head training is also not accepted full decoder training.
