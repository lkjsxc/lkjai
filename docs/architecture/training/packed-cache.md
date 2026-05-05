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

The deterministic builder is `lkjai-native-packed-cache build ...`. It loads
the local byte-level BPE `tokenizer.json`, extracts JSONL string fields named
`text` and `content` in document order, and writes fixed non-overlapping
windows. The native tokenizer is the single active tokenizer implementation.
Tokenizer-less byte or modulo mapping is forbidden for real caches.

The long-run cache path
`data/train/datasets/packed/train-causal_lm_full-seq1024` previously held a
stale smoke fixture with `sequence_len=16`, `vocab_size=256`, and no
`schema_version`. Validate this path before every seq1024 dense BF16 run; if
validation reports those fields, rebuild it with the command below.

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
- Non-smoke validation requires `tokenizer_digest`, `config_digest`,
  `source_digest`, `tokens_checksum`, `loss_mask_checksum`, `starts_checksum`,
  and `packed_data_checksum`.
- Explicit tiny smoke fixtures may set `smoke_fixture=true`; they are valid only
  for smoke gates and are rejected by real train runbooks.

## Dense Seq1024 Rebuild

```bash
docker compose --profile train run --rm --entrypoint lkjai-native-packed-cache train \
  build \
  --source /workspace/data/train/datasets/train.jsonl \
  --tokenizer /workspace/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/native_dense_20m_bf16_3070.json \
  --out /workspace/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --split train --objective causal_lm_full \
  --seq-len 1024 --sequence-count 8192 \
  --seed 20260504 --run-id dense-2h-3070
```

```bash
docker compose --profile train run --rm --entrypoint lkjai-native-packed-cache train \
  validate \
  --cache /workspace/data/train/datasets/packed/train-causal_lm_full-seq1024 \
  --source /workspace/data/train/datasets/train.jsonl \
  --tokenizer /workspace/data/train/tokenizer/tokenizer.json \
  --config /workspace/configs/native/native_dense_20m_bf16_3070.json
```
