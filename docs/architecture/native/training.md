# Native Training

## Goal

Train the scratch dense decoder without Python or PyTorch in the product path.

## Owned By Native Code

- JSONL corpus reading.
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
6. Save `lkjai-native-artifact-v1`.
7. Run fixed smoke generation against the native server.

## Acceptance

- Compose verify must pass without product Python tests.
- A synthetic GPU smoke run must complete at least two optimizer steps.
- A native artifact inspect command must validate all index offsets and shapes.
- Training speed reports median and p95 microstep time.
