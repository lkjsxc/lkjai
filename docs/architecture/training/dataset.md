# Training Dataset

Owner: `docs/architecture/training/dataset.md`.
State: canonical active dataset layout contract.

## Goal

Describe the on-disk dataset artifacts used by training and evaluation.

## Layout

- Editable source corpus: `corpus/sources/*.json`
- Public pretraining corpus:
  `data/public-corpus/{train,val,holdout}/*.jsonl`
- Accepted Kimi SFT corpus:
  `corpus/generated/kimi-sft-60m/{train,val,holdout}/*.jsonl`
- Preference pairs:
  `corpus/generated/preference-pairs/pairs/*.jsonl`
- Canonical combined corpus: `data/train/datasets/corpus.jsonl`
- Canonical train split: `data/train/datasets/train.jsonl`
- Canonical validation split: `data/train/datasets/val.jsonl`
- Canonical holdout split: `data/train/datasets/holdout.jsonl`
- Fixtures: `data/train/datasets/fixtures.jsonl`
- Metadata: `data/train/datasets/metadata.json`
- Packed cache: `data/train/datasets/packed/*/{tokens.bin,loss_mask.bin,starts.bin,metadata.json}`

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

- Packed cache metadata uses `format=lkjai-packed-cache`.
- Token ids are stored as `uint16`; the active `8192` vocabulary fits in 13
  bits.
- Loss masks remain byte masks.
- Start offsets remain unsigned 64-bit integers.
- Rebuild legacy packed caches instead of reading legacy files.

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
- Validation must fail on GPT, Claude, or generic LLM-authored default rows.
- Kimi-generated SFT rows are allowed only when runtime-contract validated,
  fixture-executed, and marked `kimi-generated`.
- Validation must fail when assistant content is not valid XML.
- Validation must fail when a resource mutation lacks prior confirmation.
- Validation must fail when a scenario family appears in multiple splits.
- Validation must fail when duplicate rows exceed 1%.
- Validation proves runtime contract and fixture shape; behavioral quality is
  still gated by model eval.

## Split Policy

- Split is scenario-family-based for generated agent SFT rows.
- Split is row-based only for legacy rows that lack a scenario family.
- Training uses `train.jsonl`.
- Validation loss uses `val.jsonl`.
- Behavioral evaluation uses `holdout.jsonl`.
- Chunked corpus readers must preserve row-based split boundaries.
