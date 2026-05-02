# Native Training

## Goal

Train native BF16 CUDA models without Python or PyTorch in the product path.
Dense remains the default production milestone. Transformer training is accepted
only through the explicit native transformer mode and only for the verified tiny
debug shape until larger runs are separately measured.

## Owned By Native Code

- JSONL corpus reading for corpus preparation.
- Prompt and target serialization.
- Byte-level BPE tokenization from `tokenizer.json`.
- Packed cache read and write.
- Pinned host batch staging.
- Dense CUDA forward, backward, AdamW, checkpoint, and export.
- Minimal native transformer CUDA training for the debug shape: token and
  learned positional embeddings, decoder layers, causal attention, SwiGLU MLP,
  norms, residuals, AdamW, checkpoints, export, and logits checks.
- Stable training reports and benchmark records.

## Data Flow

1. Read reviewed JSONL corpus rows.
2. Serialize model-facing dialogue and assistant action targets.
3. Tokenize through the native tokenizer.
4. Write or reuse `lkjai-packed-cache-v2` files.
5. Train using the selected native CUDA model kind.
6. Save `lkjai-native-artifact-v2`.
7. Probe exported single-step logits with `lkjai-native-logits-check`.

## Acceptance

- GPU-required Compose verify must pass without product Python tests.
- A native smoke run must complete at least two optimizer steps through the
  dense CUDA trainer and export a valid `lkjai-native-artifact-v2` directory.
  `lkjai-native-train --smoke` is always dense unless a future milestone states
  otherwise.
- `lkjai-native-train --train --mode transformer` is accepted only when native
  CUDA forward/backward, checkpoint/resume, BF16 export, inspect, and logits
  checks pass for the transformer debug config. The retained CPU/reference
  transformer path is not counted as completed training.
- A native artifact inspect command must validate all index offsets and shapes.
- Training persists `DATA_DIR/runs/train-report.json` and prints compact JSON
  with schema version, `model_kind`, finite decreasing loss, precision mode,
  dtype fields, phase timings, config and packed-cache digests, batch size,
  sequence length, gradient accumulation, optimizer steps, microsteps, token
  counts, BF16 export logits-check result, artifact paths/checksums, `status`,
  `failure_reason`, and `weight_changed=true`. Dense reports preserve
  `dense_cuda_path=true` for existing consumers.
- Capability reporting must show whether the run used CUDA, native BF16,
  cuBLASLt, cuDNN, and SDPA-eligible shapes.
- Chat and autoregressive decode remain out of scope. Native server
  `/v1/chat/completions` continues to return HTTP `422` with no `choices` for
  dense and transformer artifacts until decode is implemented.

## Current Implementations

- `lkjai-native-train --smoke --steps N` creates a tiny packed-cache v2 fixture,
  trains the dense BF16 CUDA embedding plus LM-head model, and exports dense
  tensors.
- The smoke export is written under `DATA_DIR/exports/${MODEL_NAME}` and
  `${DATA_DIR}/../models/${MODEL_NAME}`.
- The smoke model proves artifact load and logits inference. It is not a
  behavioral competency artifact and does not enable autoregressive chat decode.
- `lkjai-native-train --train` runs the corpus-backed native dense CUDA trainer.
  It consumes packed-cache v2 data, runs forward/backward/optimizer steps,
  checkpoints under `DATA_DIR/checkpoints`, exports under
  `DATA_DIR/exports/${MODEL_NAME}`, and mirrors the served model under
  `${DATA_DIR}/../models/${MODEL_NAME}`.
- `lkjai-native-train --train --mode transformer` runs the native transformer
  debug trainer. It uses `TRAIN_MODEL_KIND=transformer` or
  `TRAIN_CONFIG.model_kind=transformer` when the CLI flag is absent. Dense is
  still the default.
- `--resume DIR` is true dense checkpoint restore. It loads FP32 master weights
  and Adam moments from `optimizer.lkjw`, validates checkpoint compatibility,
  and rebuilds BF16 CUDA shadows before the next forward pass.
- Transformer `--resume DIR` requires a transformer checkpoint artifact with
  `optimizer.index.json` and `optimizer.lkjw`. It restores `master.*`,
  `adam_m.*`, and `adam_v.*` tensors for every trainable tensor, rebuilds BF16
  shadows, and rejects manifest kind, config, tensor shape, vocab, seed, batch,
  sequence length, gradient accumulation, and optimizer counter mismatches.
- Transformer exports write BF16 weights for every implemented trainable tensor:
  token embeddings, learned positional embeddings, per-layer RMSNorm, Q/K/V/O
  projections, SwiGLU gate/up/down projections, final norm, and LM head.
- The first transformer milestone requires `tie_embeddings=false`. Transformer
  mode fails loudly if a config requests tied embeddings. It uses correctness
  first custom CUDA kernels and FP32 accumulation; Tensor Core, cuDNN SDPA,
  cuBLASLt projection, RoPE, and larger 40M acceptance remain roadmap work until
  separately measured.
- `lkjai-native-packed-cache --migrate-v1-to-v2` wraps compatible v1 binary
  cache files as v2 after validating metadata, token width, masks, starts, vocab,
  token count, row count, file sizes, row bounds, vocab, and context
  compatibility.
- The transformer implementation remains available in source, but routine
  native training and CTests exercise the dense CUDA milestone.

## Deadline Runs

`TRAIN_CONFIG` is the training-run config. `TRAIN_NATIVE_CONFIG` or `--config`
selects the native model-shape config. Model kind precedence is:

1. CLI `--mode dense|transformer`
2. `TRAIN_MODEL_KIND=dense|transformer`
3. `TRAIN_CONFIG.model_kind=dense|transformer`
4. default `dense`

Other training values follow the existing order: CLI flags override environment
variables, environment variables override `TRAIN_CONFIG`, and the JSON config
overrides native defaults. Invalid model kinds and unsupported model/config
combinations fail before training starts.

Wall-clock stop, fixed eval, behavioral eval, autoregressive decode, automatic
tokenizer/corpus preparation, transformer performance kernels, and larger
transformer shapes remain target operations work, not current native trainer
behavior.

## Artifact Layout

Successful runs write:

- `DATA_DIR/checkpoints/latest/`
- `DATA_DIR/checkpoints/final/`
- `DATA_DIR/exports/${MODEL_NAME}/`
- `${DATA_DIR}/../models/${MODEL_NAME}/`
- `DATA_DIR/runs/train-report.json`

Checkpoint artifacts include `manifest.json`, `config.json`, `tokenizer.json`,
`weights.index.json`, `weights.lkjw`, `trainer_state.json`,
`optimizer.index.json`, and `optimizer.lkjw`. Export artifacts omit optimizer
files and are the validation target for `lkjai-native-logits-check`.

## Report Schema

Transformer reports use schema version `2` and add transformer shape fields:
`layers`, `heads`, `kv_heads`, `hidden_size`, `head_dim`, `ffn_size`, `context`,
`parameter_count`, `model_kind`, precision fields, CUDA capability, losses,
tokens/sec, artifact checksums, logits-check status, `status`, and
`failure_reason`. Dense reports retain their compatibility fields, including
`dense_cuda_path=true`, `trainer_mode`, `mode`, `steps`, `optimizer_steps`,
`tokens_seen`, and `timings`.
