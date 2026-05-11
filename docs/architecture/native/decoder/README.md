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
- [cuda-progress.md](cuda-progress.md): foundation commits, partial CUDA slice evidence,
  and remaining acceptance gap

## Boundary

| Mode | Status | Boundary |
|---|---|---|
| `dense` | Foundation | Accepted CUDA BF16 dense trainer and CUDA benchmark substrate. |
| `decoder` | Product target | Requires real tokenizer artifacts, decoder-shaped weights, native prompt serialization, and full CUDA decoder evidence before acceptance. |
| `transformer` | Reference-only | CPU/host parity source while decoder pieces are migrated; not a product training mode. |

The current decoder implementation exports real decoder artifacts with the
repo byte-level BPE tokenizer and may return partial host-reference choices
through the native tokenizer bridge. Those choices are partial usability only:
decode still calls host `transformer_forward`, the diagnostic KV writes are not
model attention state, and the route must report `lkjai_decode_supported=false`.
Training remains partial while block gradients are synthetic, full block
backward is absent, and accepted contiguous CUDA BF16 KV-cache decode is not
complete.
