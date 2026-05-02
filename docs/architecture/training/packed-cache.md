# Packed Cache Contract

## Goal

Make training consume tokenizer output, not raw JSONL, during real runs.

## Format

- Format id: `lkjai-packed-cache-v2`.
- Directory layout:
  `data/train/datasets/packed/<split>-<objective>-seq1024/`.
- Required files: `tokens.bin`, `loss_mask.bin`, `starts.bin`, and
  `metadata.json`.
- Token dtype: little-endian `uint16`.
- Loss mask dtype: byte, where `1` means the token contributes to loss.
- Start offsets dtype: little-endian `uint64`.
- Sequence length: `1024` unless the active config explicitly changes it.

## Metadata

`metadata.json` records:

- `format`,
- `split`,
- `objective`,
- `sequence_len` and `seq_len`,
- `vocab_size`,
- `token_dtype`,
- `row_count` and `sequence_count`,
- `example_count`,
- `token_count`,
- `tokenizer_digest`,
- `config_digest`,
- `source_digest`,
- `seed`,
- `run_id`,
- `max_token_id`,
- `tokens_checksum`,
- `loss_mask_checksum`,
- `starts_checksum`,
- `packed_data_checksum`.

The deterministic builder is
`cargo run -p lkjai_packed_cache_builder -- build ...`. It loads the
HuggingFace `tokenizer.json`, extracts JSONL string fields named `text` and
`content` in document order, and writes fixed non-overlapping windows. It does
not use a byte fallback or duplicate tokenizer logic in C++.

## Loader Rules

- Real non-quick native training reads packed caches by default.
- JSONL row streaming is allowed only for scaffold and corpus-construction
  commands.
- Loader code memory-maps packed files when the platform supports it.
- Dense CUDA runs stage token and mask batches through pinned host buffers and
  report H2D copy time separately from forward time. Overlap with compute is
  target optimization work.
- Bucket boundaries must be stable enough for CUDA Graph capture.

## Rebuild Rules

- Rebuild caches when tokenizer, sequence length, objective, split policy, or
  source corpus fingerprint changes.
- Do not read v1 caches in product training.
- Validation fails when `vocab_size > 65536` or when any token id exceeds the
  active tokenizer vocabulary.
- Builder validation also rejects tokenizer/config vocab mismatches,
  sequence-length/config mismatches, stale checksums, corrupt binary sizes,
  invalid fixed-window starts, and token ids outside the tokenizer or native
  config vocabulary.
