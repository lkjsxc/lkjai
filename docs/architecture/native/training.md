# Native Training

## Goal

Train the active scratch dense BF16 CUDA milestone without Python or PyTorch in
the product path.

## Owned By Native Code

- JSONL corpus reading for corpus preparation.
- Prompt and target serialization.
- Byte-level BPE tokenization from `tokenizer.json`.
- Packed cache read and write.
- Pinned host batch staging.
- Dense CUDA forward, backward, AdamW, checkpoint, and export.
- Training summaries and benchmark records.

## Data Flow

1. Read reviewed JSONL corpus rows.
2. Serialize model-facing dialogue and assistant action targets.
3. Tokenize through the native tokenizer.
4. Write or reuse `lkjai-packed-cache-v2` files.
5. Train using the native dense CUDA path.
6. Save `lkjai-native-artifact-v2`.
7. Probe exported single-step logits with `lkjai-native-logits-check`.

## Acceptance

- GPU-required Compose verify must pass without product Python tests.
- A native smoke run must complete at least two optimizer steps through the
  dense CUDA trainer and export a valid `lkjai-native-artifact-v2` directory.
- A native artifact inspect command must validate all index offsets and shapes.
- Training reports finite decreasing loss, `dense_cuda_path=true`, phase
  timings, a logits checksum, and `weight_changed=true`.
- Capability reporting must show whether the run used CUDA, native BF16,
  cuBLASLt, cuDNN, and SDPA-eligible shapes.

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
- `lkjai-native-packed-cache --migrate-v1-to-v2` wraps compatible v1 binary
  cache files as v2 after validating metadata, token width, masks, starts, vocab,
  and context compatibility.
- The transformer implementation remains available in source, but routine
  native training and CTests exercise the dense CUDA milestone.

## Deadline Runs

Use `TRAIN_STOP_AT_UNIX` for wall-clock bounded jobs. The trainer checks this
deadline every optimizer step, writes `latest`, `final`, export, fixed-eval
metadata, behavioral-eval metadata, and `checkpoints/training-summary.json`
before exiting.
