# Decoder Artifact

Owner: `docs/architecture/native/decoder/artifact.md`.
State: canonical decoder artifact contract.

## Format

Decoder exports use `lkjai-native-artifact` with `manifest.json.kind` set to
`decoder`.

## Required Files

- `manifest.json`
- `config.json`
- `tokenizer.json`
- `weights.index.json`
- `weights.lkjw`
- `trainer_state.json`
- `optimizer.index.json` and `optimizer.lkjw` for checkpoints only

`tokenizer.json` must be the local byte-level BPE tokenizer used to build the
packed cache. Decoder training/export chooses it from explicit `--tokenizer`,
`TRAIN_TOKENIZER`, training config `tokenizer`, or the repo default
`data/train/tokenizer/tokenizer.json`; missing or invalid decoder tokenizers
fail instead of writing a dummy file.

The manifest checksum covers this file and `lkjai-native-inspect` reports a
tokenizer checksum mismatch separately from config checksum mismatch. Inspect
also validates the byte-level BPE shape plus required atomic prompt/action tags,
including `<tool_name>` and `</tool_name>`.

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

Accepted decoder exports use RoPE and do not require learned
`pos_embeddings`. New decoder exports and checkpoints must not write
`pos_embeddings`; the loader's temporary tolerance is for old diagnostics only.

## Checkpoints

Decoder checkpoints add FP32 optimizer tensors:

- `master.NAME`
- `adam_m.NAME`
- `adam_v.NAME`

Resume rejects mismatched manifest kind, artifact kind, config checksum, tensor
shape, vocab, seed, batch size, sequence length, gradient accumulation, and
optimizer step metadata.

## Serving Load

The native server may return `200` from `/v1/models` only after the artifact
loads, checksums match, and tensor ranges validate. Decoder artifacts also load
and validate the tokenizer before `/v1/chat/completions` can produce choices.
Dense and transformer artifacts may still load for diagnostics, but they do
not produce chat choices.
