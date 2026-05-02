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
- `weights.lkjw` stores all model weights.
- Optional `optimizer.lkjw` stores optimizer tensors for checkpoints.
- `weights.index.json.tensors[]` entries contain `name`, `dtype`, `shape`,
  `byte_offset`, and `byte_length`.
- Supported dtypes are `u16`, `u32`, `f16`, `bf16`, and `f32`.

## Manifest Schema

`manifest.json` records:

- `format`: always `lkjai-native-artifact-v2`.
- `kind`: model family, currently `dense` for active training exports or
  `transformer` for the retained transformer artifact path.
- `artifact_kind`: `export` for serving or `checkpoint` for resume state.
- `weights_checksum`: checksum of dense tensor payloads.
- `config_checksum`: checksum of `config.json`.
- `tokenizer_checksum`: checksum of `tokenizer.json`.

The tokenizer checksum may be a placeholder while the tokenizer remains minimal,
but it must still match the bytes written in the artifact directory.

## Dense Required Tensor Names

Dense artifacts require:

- `tok_embeddings`
- `lm_head`

Dense checkpoints also require FP32 optimizer tensors for `master.*`,
`adam_m.*`, and `adam_v.*` for both dense tensors.

## Transformer Required Tensor Names

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

Training checkpoints add `optimizer.index.json` and `optimizer.lkjw`.
Optimizer tensors are FP32 master weights and Adam moments for the current dense
checkpoint. `trainer_state.json` records optimizer step, microsteps, batch size,
sequence length, gradient accumulation, loss, checksum, and checkpoint/export
kind. Scheduler counters and RNG state are target additions.

## Dense Slice

`manifest.json.kind` is `dense` for the active native BF16 CUDA milestone. The
exported weights are BF16 tensors for token embeddings and LM head. The config
records `vocab_size`, `context`, `hidden_size`, `heads`, `kv_heads`, `head_dim`,
`ffn_size`, and `seed`.

`lkjai-native-logits-check --model-dir DIR --tokens 1,2,3` loads dense
artifacts, validates finite `[1,V]` next-token logits, and emits a JSON
checksum. Transformer logits support is retained as source but is not the
accepted training milestone.

## Transformer Slice

`manifest.json.kind` is `transformer` for the retained BF16 slice. The exported
weights are BF16 tensors for token embeddings, every configured layer, final
RMSNorm, and LM head. The config records `vocab_size`, `context`, `layers`,
`hidden_size`, `heads`, `kv_heads`, `head_dim`, `ffn_size`, `activation`,
`rope_theta`, `rms_norm_eps`, `tie_embeddings`, and `seed`.

Transformer chat decode remains unsupported until the decode milestone lands.

## Compatibility

- Native artifacts do not need to load Python `model.pt` checkpoints.
- Product serving reads only `lkjai-native-artifact-v2`.
- The tokenizer remains `tokenizer.json` because that file is part of the model
  behavior contract.
