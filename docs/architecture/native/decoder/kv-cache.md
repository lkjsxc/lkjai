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
- Stop-token behavior must be tested before partial CUDA reference decode can
  be replaced in route evidence.

## Implementation Shape

- The cache layout is layer-major, then batch, KV head, position, and head dim.
- K and V tensors are native BF16 device buffers with byte offsets derived from
  the tested layout helper.
- Prefill may reuse decoder forward buffers, but steady-state decode must reuse
  cache and workspace allocations across output tokens.
- The serving response may expose accepted backend names only after CUDA write,
  append, read, and attention consumption behavior is covered by CTest and
  route contracts.
- The first accepted cache may be one contiguous per-request allocation.
  Block-pool reuse and eviction are later scheduler work, but their metrics
  must be present before continuous batching is claimed.

## Current Status

The current decoder bridge uses native CUDA reference decode and reports
`cuda_reference_kv_cache` plus partial KV-cache metadata. That is valid partial
serving disclosure, not accepted KV-cache decode.

The native implementation now has a tested layout helper for the accepted
contiguous BF16 K/V memory contract. It does not change serving reports until
decode writes model K/V tensors, reads them through CUDA attention, avoids
steady-state token allocations, and stops using host prompt recompute.
