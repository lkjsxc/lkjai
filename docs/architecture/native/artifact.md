# Native Artifact Format

## Format

The product artifact format is `lkjai-native-artifact-v1`.

Each exported model directory contains:

- `manifest.json`
- `config.json`
- `tokenizer.json`
- `weights.index.json`
- `weights.lkjw`
- `trainer_state.json`
- optional `optimizer.index.json`
- optional `optimizer.lkjw`

## Tensor Storage

- Tensor payloads are little-endian.
- Tensor payload offsets are 256-byte aligned.
- One binary file stores all model weights.
- One optional binary file stores optimizer tensors.
- Index entries contain name, dtype, shape, byte offset, and byte length.
- Supported dtypes are `u16`, `u32`, `f16`, `bf16`, and `f32`.

## Compatibility

- Native artifacts do not need to load Python `model.pt` checkpoints.
- Exporters may read older data temporarily, but product serving reads only the
  native format.
- The tokenizer remains `tokenizer.json` because that file is part of the model
  behavior contract.
