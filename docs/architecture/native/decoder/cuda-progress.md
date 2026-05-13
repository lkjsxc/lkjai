# Decoder CUDA Progress

Owner: `docs/architecture/native/decoder/cuda-progress.md`.
State: historical progress record.

## Evidence Trail

Decoder foundation landed in these commits:

- `cea2e70`: repaired source topology and docs navigation.
- `c47dd91`: defined the decoder native path and contracts.
- `2885f2b`: added decoder foundation model-kind, artifact, report, inspect/logits,
  and server routing contracts.
- `4d13332`: added the decoder end-to-end runner shell.

First CUDA progress after foundation:

- `01dac62`: adds partial CUDA BF16 decoder smoke training.
- `a806c88`: gates full two-hour decoder acceptance on a full accepted CUDA
  decoder report.
- Current change: decoder exports copy and checksum the real byte-level BPE
  tokenizer, native serving serializes XML-like chat prompts, and decoder chat
  decode uses tokenizer ids plus sampler controls.
- Current hardening: ordered `messages[]` parsing, strict sampler validation,
  tokenizer checksum and atomic-tag inspection, decoder acceptance-gate
  hardening, and standalone BF16 RMSNorm CUDA parity.
- Current forward substrate: decoder block metadata validation plus CUDA BF16
  RMSNorm, row-major cuBLASLt Q/K/V, RoPE, causal GQA attention, O projection,
  residual adds, MLP RMSNorm, BF16 SwiGLU glue, and down projection.
- Current block parity: the training-slice block forward path returns the first
  block hidden output and CTest compares it with a host reference under
  BF16-aware tolerance.
- Current attention hook: deterministic BF16 causal MHA/GQA CUDA parity plus
  reusable cuBLASLt projection plan-cache coverage.
- Current stateful forward hook: a reusable CUDA decoder layer object owns
  device-resident layer weights and intermediates, and a full-forward probe
  compares final hidden/logits against the host reference on
  `decoder_debug_bf16`.
- Current training hook: decoder training uses the full CUDA forward stack and
  CUDA CE/loss-gradient helper for loss and captured logits, while backward and
  gradient source remain host-reference. Registry-wide CUDA AdamW updates
  device FP32 masters, moments, and BF16 shadows for every decoder tensor.
  Reports name the slice `cuda_full_forward_host_backward` and remain
  non-accepted.
- Current backward substrate hook: residual-add backward and SwiGLU backward
  kernels have direct parity tests, but they are not wired into training
  reports or optimizer evidence.
- Current decode hook: decoder chat executes native CUDA prefill and
  incremental token steps, writes real BF16 K/V tensors into a contiguous device
  cache, and exposes partial non-accepted runtime disclosure fields.

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

The decoder full-forward substrate now runs inside decoder training before the
host-reference backward pass. It validates decoder metadata, launches RMSNorm,
projects Q/K/V, applies RoPE, runs causal GQA attention, projects the attention
output through O, adds the attention residual, runs MLP RMSNorm, applies
`silu(gate) * up`, projects through the down matrix, adds the final residual,
applies final RMSNorm, computes LM-head logits, and computes CE loss and
grad-logits on device. It is forward/loss evidence and does not train block
tensors.
The training-slice block test now verifies the composed first-block output
against a host reference; that is still forward correctness evidence, not
backward or optimizer acceptance.

The exported artifact remains `manifest.json.kind=decoder`, so inspect,
logits-check, and native server foundation chat contracts continue to operate on the
same decoder artifact shape.

## Current Acceptance Gap

The foundation server contract alone is not the accepted CUDA decoder trainer.
Current reports must prove:

- `implementation_status=accepted`
- `accepted_cuda_training=true`
- `decoder_cuda_slice=full_decoder`
- `decoder_block_backend=cuda_full_decoder`
- `decoder_backward_backend=cuda_full_decoder`
- `decoder_block_weight_changed=true`
- `kv_cache_backend=cuda_contiguous_bf16`
- `decode_backend=cuda_kv_cache`

Decoder chat serving may still disclose `cuda_reference_kv_cache` plus
`cuda_contiguous_bf16_partial` for artifacts without accepted route evidence.
Accepted disclosure requires the sidecar and executed CUDA KV-cache path to
agree.

Before tracked acceptance, the repo still needs full decoder CUDA backward,
accepted cuDNN SDPA attention, a real two-hour RTX acceptance run with full
decoder weight deltas, logits/export checks, route transcript, positive prefill
allocation, and zero steady-state token allocations.

## Hardware Implications

RTX 3070 remains the first acceptance target. Smoke runs prove contracts and
small-shape behavior, but they do not predict final two-hour throughput.

RTX 5090 should use separate presets and autotune data after the RTX 3070 lane
has accepted evidence. Smoke builds there are not enough evidence for Blackwell
decoder performance.

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
passed. The two-hour acceptance job remains the required generated evidence for
the full accepted CUDA decoder backend.
