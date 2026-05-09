# Native Training

## Goal

Train native BF16 CUDA models without Python or PyTorch in the product path.
The current accepted CUDA training path is dense BF16 CUDA only. The next
accepted target is the `decoder` model kind. Transformer mode remains available
for reference plumbing, but it is experimental until it is replaced or retired.

## Owned By Native Code

- JSONL corpus reading for corpus preparation.
- Prompt and target serialization.
- Byte-level BPE tokenization from `tokenizer.json`.
- Packed cache read and write.
- Pinned host batch staging.
- Dense CUDA forward, backward, AdamW, checkpoint, and export.
- Persistent dense packed-cache reads after one run-start validation pass.
- Reusable dense CUDA step buffers and cached cuBLASLt plans for steady shapes.
- Experimental transformer host/reference training with a CUDA capability
  probe, checkpoints, export, and logits checks.
- Target decoder CUDA training with BF16 weights, FP32 optimizer state, native
  deadline stop, export, and decode checks.
- Stable training reports and benchmark records.

## Data Flow

1. Read reviewed JSONL corpus rows.
2. Serialize model-facing dialogue and assistant action targets.
3. Tokenize through the native tokenizer.
4. Write or reuse `lkjai-packed-cache` files.
5. Train using the selected native CUDA model kind.
6. Save `lkjai-native-artifact`.
7. Probe exported single-step logits with `lkjai-native-infer`.
8. Use `lkjai-native-logits-check` only for validation/reference checks.

## Acceptance

- GPU-required Compose verify must pass without product Python tests.
- A native smoke run must complete at least two optimizer steps through the
  dense CUDA trainer and export a valid `lkjai-native-artifact` directory.
  `lkjai-native-train --smoke` is always dense unless a future target states
  otherwise.
- `lkjai-native-train --train --mode transformer` must remain callable, but its
  report must declare `accepted_cuda_training=false`,
  `implementation_status=experimental`, and host/reference backends. It is not
  accepted CUDA training.
- A native artifact inspect command must validate all index offsets and shapes.
- Training persists `DATA_DIR/runs/train-report.json` and prints compact JSON
  with schema identifier, `model_kind`, finite decreasing loss, precision mode,
  dtype fields, phase timings, config and packed-cache digests, batch size,
  sequence length, gradient accumulation, optimizer steps, microsteps, token
  counts, BF16 export logits-check result, artifact paths/checksums, `status`,
  `failure_reason`, and `weight_changed=true`.
- Capability reporting must show whether the run used CUDA, native BF16,
  cuBLASLt, cuDNN, device memory, SM count, async allocation support, build
  architecture flags, and SDPA device/library eligibility.
- Dense reports must state the persistent packed-cache reader, physical `B*S`
  row layout with masked final-token loss, cuBLASLt plan cache, reusable step
  buffers, dense backward GEMM/scatter status, step-buffer byte counts, and
  CUDA-event timing source.
- Chat and autoregressive decode remain out of scope for dense and transformer
  artifacts. Native server `/v1/chat/completions` returns HTTP `422` with no
  `choices` for those kinds. Decoder artifacts with the real local tokenizer
  may return `choices`, even while full CUDA decoder training remains partial.

## Current Implementations

- `lkjai-native-train --smoke --steps N` creates a tiny packed-cache fixture,
  trains the dense BF16 CUDA embedding plus LM-head model, and exports dense
  tensors.
- The smoke export is written under `DATA_DIR/exports/${MODEL_NAME}` and
  `${DATA_DIR}/../models/${MODEL_NAME}`.
- The smoke model proves artifact load and logits inference. It is not a
  behavioral competency artifact and does not enable autoregressive chat decode.
- `lkjai-native-train --train` runs the corpus-backed native dense CUDA trainer.
  It consumes packed-cache data, runs forward/backward/optimizer steps,
  checkpoints under `DATA_DIR/checkpoints`, exports under
  `DATA_DIR/exports/${MODEL_NAME}`, and mirrors the served model under
  `${DATA_DIR}/../models/${MODEL_NAME}`.
- `lkjai-native-infer --model-dir DIR --tokens CSV` loads dense BF16 exports
  and emits stable logits JSON. It rejects missing, corrupt, transformer, and
  out-of-range token inputs. It does not decode text.
- `lkjai-native-train --train --mode transformer` runs the experimental
  transformer debug trainer. It uses `TRAIN_MODEL_KIND=transformer` or
  `TRAIN_CONFIG.model_kind=transformer` when the CLI flag is absent. Dense is
  still the default and the only accepted CUDA training mode.
- `--resume DIR` is true dense checkpoint restore. It loads FP32 master weights
  and Adam moments from `optimizer.lkjw`, validates checkpoint match,
  and rebuilds BF16 CUDA shadows before the next forward pass.
- Experimental transformer `--resume DIR` requires a transformer checkpoint with
  `optimizer.index.json` and `optimizer.lkjw`. It restores `master.*`,
  `adam_m.*`, and `adam_v.*` tensors for every trainable tensor, rebuilds BF16
  shadows, and rejects manifest kind, config, tensor shape, vocab, seed, batch,
  sequence length, gradient accumulation, and optimizer counter mismatches.
- Transformer exports write BF16 weights for every implemented trainable tensor:
  token embeddings, learned positional embeddings, per-layer RMSNorm, Q/K/V/O
  projections, SwiGLU gate/up/down projections, final norm, and LM head.
- The first transformer target requires `tie_embeddings=false`. Transformer
  mode fails loudly if a config requests tied embeddings. It currently reports
  `forward_backend=host_reference`, `backward_backend=host_surrogate`,
  `optimizer_backend=host_adamw_fp32`, and `transformer_cuda_probe=true`.
  Tensor Core, cuDNN SDPA, cuBLASLt transformer projection, RoPE, and accepted
  transformer CUDA training remain backlog work until real kernels replace the
  reference path.
- Legacy binary cache migration is removed from the product path. Rebuild caches
  with `lkjai-native-packed-cache build` after tokenizer, source, objective, or
  sequence-length changes.
- The transformer implementation remains available in source, but routine
  native training and CTests exercise the dense CUDA target.

## Deadline Runs

`TRAIN_CONFIG` is the training-run config. `TRAIN_NATIVE_CONFIG` or `--config`
selects the native model-shape config. Model kind precedence is:

1. CLI `--mode dense|transformer|decoder`
2. `TRAIN_MODEL_KIND=dense|transformer|decoder`
3. `TRAIN_CONFIG.model_kind=dense|transformer|decoder`
4. default `dense`

Other training values follow the existing order: CLI flags override environment
variables, environment variables override `TRAIN_CONFIG`, and the JSON config
overrides native defaults. Invalid model kinds and unsupported model/config
combinations fail before training starts.

Wall-clock stop is implemented for the current partial decoder slice and is
target behavior for accepted decoder runs. It is not an accepted dense training
gate. Fixed eval, behavioral eval, automatic tokenizer/corpus preparation,
transformer performance kernels, and larger transformer shapes remain target
operations work.

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
files and are the validation target for dense infer and logits checks.

## Report Schema

All native train reports use stable schema. Common fields include
`model_kind`, `accepted_cuda_training`, `implementation_status`,
`forward_backend`, `backward_backend`, `optimizer_backend`,
`cuda_probe_passed`, precision fields, `limitations`, artifact paths/checksums,
losses, timings, and capability. `timings.h2d` is separate from
`timings.forward`.

Stable schema capability fields are additive. Consumers must preserve existing
fields and tolerate the hardware/build fields listed in
[capability.md](capability.md).

Dense reports declare `accepted_cuda_training=true`,
`implementation_status=accepted`, `dense_cuda_path=true`,
`forward_backend=cuda_bf16_cublaslt`,
`backward_backend=cuda_bf16_cublaslt_scatter`,
`backward_gemm_enabled=true`,
`embedding_grad_backend=token_scatter_add_fp32`, and
`optimizer_backend=cuda_adamw_fp32`. They also declare
`loader_backend=persistent_packed_cache_reader`,
`row_layout=dense_physical_bxseq_masked_final_token`,
`matmul_plan_cache_enabled=true`, `buffer_reuse_enabled=true`, and
`timing_source=cuda_events_with_boundary_sync`. Dense reports include logits,
grad-logits, hidden-gradient, and cuBLASLt workspace byte counts as additive
stable-schema fields. Dense logits checks compare BF16 exports against FP32
checkpoint masters when a reference checkpoint is available.

Decoder reports declare `model_kind=decoder` and use additive fields from
[decoder/training.md](decoder/training.md), including
`decoder_block_weight_changed` so LM-head-only updates cannot satisfy decoder
block-training acceptance.

Transformer reports declare `accepted_cuda_training=false`,
`implementation_status=experimental`, `transformer_status=experimental`,
`transformer_cuda_path=false`, `transformer_cuda_probe=true`,
`forward_backend=host_reference`, `backward_backend=host_surrogate`, and
`optimizer_backend=host_adamw_fp32`.
