# Native Artifact Format

## Format

The product artifact format is `lkjai-native-artifact`.

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

- `format`: always `lkjai-native-artifact`.
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
`adam_m.*`, and `adam_v.*` for both dense tensors. Resume restores these FP32
master tensors and Adam moments, then rebuilds the CUDA BF16 shadow tensors from
the restored masters before training continues.

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
kind. `--resume DIR` rejects mismatched manifest/config/model shape, vocab,
seed, batch, sequence, gradient accumulation, or dense tensor shape instead of
falling back to step-only resume. Scheduler counters, retained checkpoint
history, and best-metric promotion are target additions.

## Dense Slice

`manifest.json.kind` is `dense` for the active native BF16 CUDA target. The
exported weights are BF16 tensors for token embeddings and LM head. The config
records `vocab_size`, `context`, `hidden_size`, `heads`, `kv_heads`, `head_dim`,
`ffn_size`, and `seed`.

`lkjai-native-logits-check --model-dir DIR --tokens 1,2,3` loads exported BF16
dense weights, validates finite `[1,V]` next-token logits, and emits a JSON
checksum. Its tolerance and checksum contract validate BF16 export behavior, not
FP32-master parity. Transformer logits support is retained as source but is not
the accepted training target.

## Transformer Slice

`manifest.json.kind` is `transformer` for the retained BF16 slice. The exported
weights are BF16 tensors for token embeddings, every configured layer, final
RMSNorm, and LM head. The config records `vocab_size`, `context`, `layers`,
`hidden_size`, `heads`, `kv_heads`, `head_dim`, `ffn_size`, `activation`,
`rope_theta`, `rms_norm_eps`, `tie_embeddings`, and `seed`.

Transformer chat decode remains unsupported. Decoder artifacts own chat-capable
serving.

## Compatibility

- Native artifacts do not need to load Python `model.pt` checkpoints.
- Product serving reads only `lkjai-native-artifact`.
- The tokenizer remains `tokenizer.json` because that file is part of the model
  behavior contract.
