# Decoder Inference Session

Owner: `docs/architecture/native/decoder/inference-session.md`.
State: partial implementation contract.

## Target

Decoder serving owns a cached CUDA inference session per loaded decoder
artifact. The session preloads artifact weights once, owns reusable execution
context and workspace objects, and runs prompt prefill plus one-token decode
against a per-request KV cache.

## Required Behavior

- Artifact loading constructs `DecoderCudaInferenceSession` once for the cached
  artifact.
- Request handling allocates a fresh contiguous BF16 KV cache for the request.
- Session construction preloads token embeddings, final norm, LM-head, and
  layer weights to device memory.
- Generation reports measured device allocation deltas from the device
  substrate, not hand-authored allocation counters.
- Partial decode may return generated choices, but it must keep
  `lkjai_decode_accepted=false` and partial backend names until the accepted
  report, sidecar, model shape, route, prefill bytes, and zero steady-state
  allocation gates pass.

## Allocation Accounting

The device substrate counts `DeviceTensor`, `DeviceWorkspace`, temporary BF16
conversion buffers, and decoder KV-cache allocations. Decode reports use those
counters to derive steady-state token allocation counts.

Accepted decode requires:

- positive KV prefill bytes,
- zero measured allocations in the steady one-token decode loop,
- CUDA KV-cache use on the executed route,
- no accepted backend names from sidecar metadata alone.

The current partial route proves prefill allocation and per-request session
lifecycle. It is not accepted until multi-token steady-state reuse and accepted
training evidence satisfy the training and route report gates.
