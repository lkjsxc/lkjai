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

The RMSNorm CUDA slice is standalone: BF16 input/output, FP32 weight, FP32
sum-of-squares reduction, and one row per CUDA block. It has parity coverage
against a CPU reference but is not wired into decoder block training.

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
RMSNorm integration and backward coverage, RoPE, attention or GQA, SwiGLU MLP,
full optimizer coverage for block tensors, contiguous BF16 KV-cache decode, no
per-token device allocations, and a real two-hour RTX acceptance run.

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
