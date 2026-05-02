# Native Training

## Goal

Train the scratch dense decoder without Python or PyTorch in the product path.

## Owned By Native Code

- JSONL corpus reading for corpus preparation.
- Prompt and target serialization.
- Byte-level BPE tokenization from `tokenizer.json`.
- Packed cache read and write.
- Pinned host batch staging.
- Forward, backward, optimizer, checkpoint, and export.
- Training summaries and benchmark records.

## Data Flow

1. Read reviewed JSONL corpus rows.
2. Serialize model-facing dialogue and assistant action targets.
3. Tokenize through the native tokenizer.
4. Write or reuse `lkjai-packed-cache-v2` files.
5. Train using native C++/CUDA kernels and vendor libraries.
6. Save `lkjai-native-artifact-v2`.
7. Run fixed smoke generation against the native server.

## Acceptance

- GPU-required Compose verify must pass without product Python tests.
- A native smoke run must complete at least two optimizer steps and export a
  valid `lkjai-native-artifact-v2` directory.
- A native artifact inspect command must validate all index offsets and shapes.
- Training speed reports median and p95 microstep time.

## Current Implementations

- `lkjai-native-train --smoke --steps N` runs a tiny dense native training loop
  and exports named dense tensors.
- The smoke export is written under `DATA_DIR/exports/${MODEL_NAME}` and
  `${DATA_DIR}/../models/${MODEL_NAME}`.
- The smoke model proves the artifact, load, and decode path. It is not a
  behavioral competency artifact.
- `lkjai-native-train --train` runs the corpus-backed dense native trainer. It
  consumes packed-cache v2 data, runs forward/backward/optimizer steps,
  checkpoints under `DATA_DIR/checkpoints`, exports under
  `DATA_DIR/exports/${MODEL_NAME}`, and mirrors the served model under
  `${DATA_DIR}/../models/${MODEL_NAME}`.

## Deadline Runs

Use `TRAIN_STOP_AT_UNIX` for wall-clock bounded jobs. The trainer checks this
deadline every optimizer step, writes `latest`, `final`, export, fixed-eval
metadata, behavioral-eval metadata, and `checkpoints/training-summary.json`
before exiting.
