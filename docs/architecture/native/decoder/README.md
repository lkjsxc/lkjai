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

`decoder` is the target chat-capable path. `dense` remains the accepted BF16
CUDA foundation until `decoder` has its own evidence. The retained
`transformer` path is experimental host/reference plumbing and must not be
promoted as accepted CUDA training.
