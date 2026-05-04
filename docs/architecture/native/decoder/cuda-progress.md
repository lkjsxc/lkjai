# Decoder CUDA Progress

## Evidence Trail

Decoder P0 landed in these commits:

- `cea2e70`: repaired source topology and docs navigation.
- `c47dd91`: defined the decoder native path and contracts.
- `2885f2b`: added decoder P0 model-kind, artifact, report, inspect/logits,
  and server routing contracts.
- `4d13332`: added the decoder end-to-end runner shell.

First CUDA progress after P0:

- `01dac62`: adds partial CUDA BF16 decoder smoke training.
- `a806c88`: gates full two-hour decoder acceptance on a full accepted CUDA
  decoder report.

## What Is CUDA-Backed

Commit `01dac62` trains the decoder token embeddings and LM head through the
existing dense CUDA substrate:

- BF16 parameter shadows on device.
- FP32 master weights on device.
- FP32 Adam/AdamW moment state on device.
- cuBLASLt GEMMs for logits and gradients.
- CUDA kernels for BF16/FP32 casts, CE loss/gradient, scatter-add embedding
  gradients, and AdamW updates.
- Reusable CUDA workspace and report fields for workspace usage.

The exported artifact remains `manifest.json.kind=decoder`, so inspect,
logits-check, and native server P0 chat contracts continue to operate on the
same decoder artifact shape.

## What Is Not Accepted Yet

P0 server contract is not the accepted CUDA decoder trainer.

The partial CUDA slice is also not the accepted CUDA decoder trainer. Reports
must say:

- `implementation_status=partial_cuda`
- `accepted_cuda_training=false`
- `decoder_cuda_slice=embedding_lm_head`
- `decoder_block_backend=static_reference`
- `attention_backend=not_implemented`

Before acceptance, the repo still needs CUDA decoder block forward/backward,
RMSNorm, RoPE, attention or GQA, SwiGLU MLP, full optimizer coverage for block
tensors, KV-cache-backed decode, tokenizer-owned prompt handling, and a real
two-hour RTX acceptance run.

## Hardware Implications

RTX 3070 remains the first acceptance target. The current slice is useful
because it proves the decoder artifact can be trained through the device
resident BF16/FP32 substrate without changing server contracts, but it is much
cheaper than full decoder blocks and does not predict final two-hour throughput.

RTX 5090 should use separate presets and autotune data once full decoder blocks
exist. The current slice should build there, but it is not enough evidence for
Blackwell decoder performance.

## Verification

Verified command:

```bash
docker compose --progress quiet --profile verify run --rm verify
```

Result on this change set: pass. The actual two-hour acceptance job was not
run, because a full accepted CUDA decoder backend does not exist yet.
