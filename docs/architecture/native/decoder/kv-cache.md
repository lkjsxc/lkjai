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
- Stop-token behavior must stay covered before accepted CUDA KV-cache decode is
  used in route evidence.

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

The decoder bridge can disclose either accepted CUDA KV-cache decode or
`cuda_reference_kv_cache` plus partial KV-cache metadata. The accepted name is
allowed only when the sidecar and executed route path agree.

The native implementation has partial contiguous BF16 K/V allocation evidence,
CUDA append plumbing, allocation counters, and route disclosure fields. This is
not accepted KV-cache serving yet. Acceptance still requires proving prompt
prefill allocation is positive, token-loop device allocations stay at zero, and
the decode path consumes the cache without full-prompt host recompute.

Future batching work must preserve zero steady-state token allocations.
