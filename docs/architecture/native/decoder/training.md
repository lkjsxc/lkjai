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
- `decoder_cuda_path`
- `attention_backend`
- `matmul_backend`
- `optimizer_backend`
- `decode_supported`
- `target_seconds`
- `deadline_hit`
- `stop_reason`
- `kv_cache_backend`

Reports are accepted only after finite loss, nonzero weight change,
checkpoint/export validation, logits checks, server decode checks, and the
documented benchmark gate pass.
