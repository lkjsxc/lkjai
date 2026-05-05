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
- Current change: decoder exports copy and checksum the real byte-level BPE
  tokenizer, native serving serializes XML-like chat prompts, and decoder chat
  decode uses tokenizer ids plus sampler controls.
- Current hardening: ordered `messages[]` parsing, strict sampler validation,
  tokenizer checksum and atomic-tag inspection, decoder acceptance-gate
  hardening, and standalone BF16 RMSNorm CUDA parity.
- Current forward substrate: decoder block metadata validation plus a CUDA
  probe for BF16 RMSNorm, BF16 RoPE, row-major BF16 cuBLASLt Q/K/V/O
  projections, and BF16 SwiGLU glue.
- Current attention hook: deterministic BF16 causal GQA CUDA parity plus
  reusable cuBLASLt projection plan-cache coverage.

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

The decoder block forward substrate is standalone and probed before the
existing decoder CUDA slice runs. It validates decoder metadata, launches the
RMSNorm primitive, applies RoPE to Q/K tensors, probes Q/K/V/O projection
GEMMs through cuBLASLt, and runs `silu(gate) * up`. It is forward-only evidence
and does not train block tensors.

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
- `decoder_block_backend=cuda_forward_partial`
- `rmsnorm_backend=cuda_bf16_fp32_reduce`
- `rope_backend=cuda_bf16`
- `qkv_projection_backend=cuda_bf16_cublaslt`
- `attention_backend=not_implemented` in trainer reports
- `mlp_backend=cuda_swiglu_partial`
- `decoder_backward_backend=not_implemented`
- `kv_cache_backend=none`
- `decode_backend=host_reference_recompute`

Decoder chat serving is also partial. Successful decoder `choices` responses
must disclose `lkjai_decode_backend=host_reference_recompute` and
`lkjai_kv_cache_backend=none` until the accepted contiguous BF16 KV-cache path
lands.

Before acceptance, the repo still needs the CUDA attention hook wired into full
decoder forward, full block backward, down projection and optimizer coverage for
block tensors, contiguous BF16 KV-cache decode, no per-token device allocations,
and a real two-hour RTX acceptance run.

## Hardware Implications

RTX 3070 remains the first acceptance target. The current slice is useful
because it proves the decoder artifact can be trained through the device
resident BF16/FP32 substrate without changing server contracts, but it is much
cheaper than full decoder blocks and does not predict final two-hour throughput.

RTX 5090 should use separate presets and autotune data once full decoder blocks
exist. The current slice should build there, but it is not enough evidence for
Blackwell decoder performance.

## Verification

Verification commands for this change set:

```bash
cmake -S native -B /tmp/lkjai-native-build -G Ninja
cmake --build /tmp/lkjai-native-build --parallel
ctest --test-dir /tmp/lkjai-native-build --output-on-failure
docker compose --progress quiet --profile verify run --build --rm verify
```

Actual result on this change set: the local shell did not have `cmake` or
`ninja`, so direct host configure/build commands were not runnable. Docker
verify ran native configure, native build, and CTest inside the verify image and
passed. The two-hour acceptance job is still not applicable because a full
accepted CUDA decoder backend does not exist yet.
