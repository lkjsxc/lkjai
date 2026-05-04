# Decoder Artifact

## Format

Decoder exports use `lkjai-native-artifact-v2` with `manifest.json.kind` set to
`decoder`.

## Required Files

- `manifest.json`
- `config.json`
- `tokenizer.json`
- `weights.index.json`
- `weights.lkjw`
- `trainer_state.json`
- `optimizer.index.json` and `optimizer.lkjw` for checkpoints only

## Required Weight Tensors

- `tok_embeddings`
- `layers.N.attn_norm`
- `layers.N.attn.q_proj`
- `layers.N.attn.k_proj`
- `layers.N.attn.v_proj`
- `layers.N.attn.o_proj`
- `layers.N.mlp_norm`
- `layers.N.mlp.gate_proj`
- `layers.N.mlp.up_proj`
- `layers.N.mlp.down_proj`
- `final_norm`
- `lm_head`

`N` is zero-based and must match `config.json.layers`.

## Checkpoints

Decoder checkpoints add FP32 optimizer tensors:

- `master.NAME`
- `adam_m.NAME`
- `adam_v.NAME`

Resume rejects mismatched manifest kind, artifact kind, config checksum, tensor
shape, vocab, seed, batch size, sequence length, gradient accumulation, and
optimizer step metadata.

## Serving Load

The native server may return `200` from `/v1/models` only after the decoder
artifact loads, the tokenizer loads, and all required tensor ranges validate.
Dense and transformer artifacts may still load for readiness checks, but only
decoder artifacts can produce chat choices.
