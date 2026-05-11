# Device Tensor Contract

`DeviceTensor` is the canonical native storage substrate for CUDA-resident
model tensors and temporary buffers.

## Ownership

- A tensor owns one contiguous device allocation.
- Shape and dtype travel with the allocation.
- Empty shapes own no allocation and report zero bytes.
- Moves transfer ownership; copies are disabled.
- Destruction releases the allocation through the same allocation policy that
  created it.

## Dtypes

The initial device substrate supports:

- `f32`: host-visible reference and reduction format.
- `bf16`: product forward/training storage format.

Later dtypes must extend the dtype enum, byte-size helper, and round-trip tests
before artifact or kernel code can depend on them.

## Stream Behavior

- Host-to-device and device-to-host copies accept an explicit stream.
- Synchronous helpers are allowed for tests and command-line smoke tools.
- BF16 conversion kernels run on the copy stream and synchronize only when the
  synchronous helper contract requires host visibility.
- Future forward kernels must reuse the caller-owned execution context stream.

## Workspace

Reusable scratch memory is owned by a small workspace allocator. It uses CUDA
async allocation when the active device reports pool support, and falls back to
ordinary `cudaMalloc`/`cudaFree` when unavailable.

The workspace is a foundation utility only; it must not hide persistent model
weights or KV-cache ownership.
