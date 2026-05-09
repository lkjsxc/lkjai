# Decoder KV Cache

Owner: `docs/architecture/native/decoder/kv-cache.md`.
State: future acceptance contract.

## Acceptance Target

Accepted decode uses a native-owned contiguous BF16 KV cache for incremental
autoregressive generation.

## Required Behavior

- Prefill writes K/V tensors for the prompt once.
- Each decode step appends one token's K/V tensors without recomputing the full
  prompt.
- Steady-state decode does not allocate device memory per token.
- Reports and responses name the backend, cache dtype, and unsupported modes.
- Accepted reports use `kv_cache_backend=cuda_contiguous_bf16` and
  `decode_backend=cuda_kv_cache`.
- Decode reports must include cache allocation accounting and prove zero
  steady-state device allocations per generated token.
- Stop-token behavior must be tested before host-reference recompute decode can
  be replaced in route evidence.

## Implementation Shape

- The cache layout is layer-major, then batch, KV head, position, and head dim.
- K and V tensors share one native allocation range with byte offsets derived
  from the tested layout helper.
- Prefill may reuse decoder forward buffers, but steady-state decode must reuse
  cache and workspace allocations across output tokens.
- The serving response may expose accepted backend names only after CUDA write,
  append, and read behavior is covered by CTest and route contracts.

## Current Status

The current decoder bridge recomputes the host reference each token and reports
`host_reference_recompute` plus `kv_cache_backend=none`. That is valid partial
serving evidence, not accepted KV-cache decode.

The native implementation now has a tested layout helper for the accepted
contiguous BF16 K/V memory contract. It does not change serving reports until
decode writes and reads the cache on the CUDA path.
