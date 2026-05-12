# Decoder Native Path

Use this subtree for the native `decoder` product target.

## Read This Section When

- You need the same-model train-to-chat path.
- You need decoder artifact, training, or decode contracts.
- You need RTX 3070 and RTX 5090 decoder presets.
- You need the two-hour decoder benchmark acceptance rules.

## Child Index

- [config.md](config.md): model-kind, shape, precision, and preset defaults
- [artifact.md](artifact.md): decoder artifact tensors and manifest behavior
- [training.md](training.md): CUDA training ownership, wall-clock stop, and reports
- [decode.md](decode.md): native autoregressive decode, sampler, KV cache, and API
- [attention.md](attention.md): full attention acceptance requirements
- [backward.md](backward.md): decoder backward and optimizer acceptance
- [kv-cache.md](kv-cache.md): accepted contiguous BF16 KV-cache decode contract
- [benchmark.md](benchmark.md): smoke, two-hour, and evidence requirements
- [cuda-progress.md](cuda-progress.md): foundation commits, accepted decoder path,
  and remaining evidence gap

## Boundary

| Mode | Status | Boundary |
|---|---|---|
| `dense` | Foundation | Accepted CUDA BF16 dense trainer and CUDA benchmark substrate. |
| `decoder` | Product target | Requires real tokenizer artifacts, decoder-shaped weights, native prompt serialization, full CUDA decoder evidence, and accepted route evidence. |
| `transformer` | Reference-only | CPU/host parity source while decoder pieces are migrated; not a product training mode. |

The decoder implementation exports decoder artifacts with the repo byte-level
BPE tokenizer and returns CUDA choices through the native tokenizer bridge.
Accepted artifacts report `lkjai_decode_backend=cuda_kv_cache` and
`lkjai_kv_cache_backend=cuda_contiguous_bf16`; artifacts without accepted route
evidence report the explicit non-accepted CUDA reference names.
