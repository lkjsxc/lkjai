# Training Dataset

## Goal

Describe the on-disk dataset artifacts used by training and evaluation.

## Layout

- Editable source corpus: `corpus/sources/*.json`
- Public pretraining corpus:
  `data/public-corpus/{train,val,holdout}/*.jsonl`
- Historical Kimi corpus:
  `corpus/generated/kimi-full-v1/{train,val,holdout}/*.jsonl`
- Canonical combined corpus: `data/train/datasets/corpus.jsonl`
- Canonical train split: `data/train/datasets/train.jsonl`
- Canonical validation split: `data/train/datasets/val.jsonl`
- Canonical holdout split: `data/train/datasets/holdout.jsonl`
- Fixtures: `data/train/datasets/fixtures.jsonl`
- Metadata: `data/train/datasets/metadata.json`
- Packed cache v2: `data/train/datasets/packed/*/{tokens.bin,loss_mask.bin,starts.bin,metadata.json}`

## Metadata

- `schema`: active schema id.
- `rows`: total rows written.
- `split_rows`: counts by split.
- `unique_rows`: normalized unique row count.
- `duplicate_rows`: normalized duplicate row count.
- `sources`: ordered source list with license and provenance details.
- `field_policy`: public pretraining field policy; active value is `text-only`.
- `excluded_fields`: public fields that must not appear in emitted rows.
- `token_budget`: optional object with `train_tokens`, `parameter_count`,
  `tokens_per_parameter`, and `chinchilla_gap`.

## Packed Cache

- Packed cache metadata uses `format=lkjai-packed-cache-v2`.
- Token ids are stored as `uint16`; the active `8192` vocabulary fits in 13
  bits.
- Loss masks remain byte masks.
- Start offsets remain unsigned 64-bit integers.
- Rebuild old packed caches instead of reading v1 files.

## Validation

- Source validation requires each JSON source entry to contain `tags` and
  object-shaped `content`.
- Validation requires at least one row in every emitted split file.
- Chunked corpus validation requires each non-final chunk to contain roughly
  `1000` lines.
- SFT rows must contain valid `messages`, `tags`, and `meta`.
- Pretraining rows must contain `mode=pretrain`, English `text`, and
  source/license metadata.
- Public pretraining rows must not include source `prompt` or `seed_data`
  values.
- Validation must fail on missing split labels or missing provenance fields.
- Validation must fail on GPT, Kimi, Claude, or generic LLM-authored default
  rows.
- Validation must fail when assistant content is not valid XML.
- Validation must fail when duplicate rows exceed 1%.
- Validation proves shape only, not quality.

## Split Policy

- Split is row-based, not token-stream-based.
- Training uses `train.jsonl`.
- Validation loss uses `val.jsonl`.
- Behavioral evaluation uses `holdout.jsonl`.
- Chunked corpus readers must preserve row-based split boundaries.
