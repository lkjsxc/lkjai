# Dense Decoder Native Contract

## Goal

Use a real dense decoder for native training and serving while keeping the
native artifact, runtime, and verification boundaries stable.

## Active Shape

- Preset: `scratch-40m`.
- Vocabulary: `8192`.
- Context: `1024`.
- Layers: `10`.
- Hidden size: `576`.
- Attention heads: `8`.
- KV heads: `2`.
- Head dimension: `72`.
- FFN size: `1536`.
- Default precision: BF16 activations and weights with FP32 accumulation.
- Fallback precision: FP16 with native loss scaling.

## Library Ownership

- Embeddings and output projection are native tensors in `weights.lkjw`.
- QKV, output, and FFN projections use cuBLASLt first.
- Attention uses cuDNN graph or fused attention paths first.
- CUTLASS is the fallback for exact-shape kernels or custom epilogues.
- Custom CUDA owns only measured glue: RMSNorm, RoPE, KV-cache update,
  sampler, stop detection, and small fusions.

## Training State

- Optimizer: AdamW.
- Optimizer state: FP32 master weights and FP32 moments.
- Checkpoints include model tensors, optimizer tensors, scheduler counters,
  AMP scaler when FP16 is active, RNG state, validation history, and config.
- Exported serving artifacts omit optimizer tensors unless explicitly requested.

## Acceptance Milestone

The dense path is accepted only when all of these are true:

- A fixed synthetic batch trains forward and backward without NaN loss.
- CPU reference or deterministic debug math agrees within the documented
  tolerance for logits and one optimizer step.
- Packed-cache training consumes `lkjai-packed-cache-v2`.
- Native server generation uses the request messages and existing KV cache.
- Decode failure returns a non-success response; it never emits a canned action.
- GPU-required Compose verify passes on the native dense smoke.

## Non-Goals

- Do not write custom GEMM before measuring cuBLASLt.
- Do not add tensor parallelism before single-GPU acceptance.
- Do not support Python `model.pt` as a product serving artifact.
- Do not keep transition-table generation in the product path.
