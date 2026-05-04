# Decoder Native Path

Use this subtree for the chat-capable native `decoder` model kind.

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
- [benchmark.md](benchmark.md): smoke, two-hour, and evidence requirements
- [cuda-progress.md](cuda-progress.md): P0 commits, partial CUDA slice evidence,
  and remaining acceptance gap

## Boundary

| Mode | Status | Boundary |
|---|---|---|
| `dense` | Foundation | Accepted CUDA BF16 dense trainer and CUDA benchmark substrate. |
| `decoder` | Product target | Requires real tokenizer artifacts, decoder-shaped weights, native prompt serialization, and full CUDA decoder evidence before acceptance. |
| `transformer` | Reference-only | CPU/host parity source while decoder pieces are migrated; not a product training mode. |

The current decoder implementation exports real decoder artifacts with the
repo byte-level BPE tokenizer and can serve decoder chat choices through the
native tokenizer bridge. It now has standalone RMSNorm CUDA parity coverage,
but still reports partial CUDA training because attention, MLP, norm
integration, RoPE, and full backward kernels are not complete.
