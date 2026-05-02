# Native Artifact Format

## Format

The product artifact format is `lkjai-native-artifact-v2`.

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

## Required Tensor Names

- `tok_embeddings`
- `layers.N.attn.q_proj`
- `layers.N.attn.k_proj`
- `layers.N.attn.v_proj`
- `layers.N.attn.o_proj`
- `layers.N.mlp.gate_proj`
- `layers.N.mlp.up_proj`
- `layers.N.mlp.down_proj`
- `layers.N.attn_norm`
- `layers.N.mlp_norm`
- `final_norm`
- `lm_head`

`N` is zero-based and must match `config.json.layers`.

## Training State

Training checkpoints may add `optimizer.index.json` and `optimizer.lkjw`.
Optimizer tensors are reserved for FP32 master weights, Adam moments, scheduler
counters, RNG state, and resume metadata. Serving exports omit optimizer tensors
by default.

## Transformer Slice

`manifest.json.kind` is `transformer` for the native BF16 slice. The exported
weights are BF16 tensors for token embeddings, every configured layer, final
RMSNorm, and LM head. The config records `vocab_size`, `context`, `layers`,
`hidden_size`, `heads`, `kv_heads`, `head_dim`, `ffn_size`, `activation`,
`rope_theta`, `rms_norm_eps`, `tie_embeddings`, and `seed`.

`lkjai-native-logits-check --model-dir DIR --tokens 1,2,3` loads those tensors,
runs the transformer forward path, validates finite `[1,V]` next-token logits,
and emits a JSON checksum.

## Compatibility

- Native artifacts do not need to load Python `model.pt` checkpoints.
- Product serving reads only `lkjai-native-artifact-v2`.
- The tokenizer remains `tokenizer.json` because that file is part of the model
  behavior contract.
