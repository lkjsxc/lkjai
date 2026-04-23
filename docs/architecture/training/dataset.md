# Training Dataset

## Goal

Describe the on-disk dataset artifacts used by training and evaluation.

## Layout

- Editable source corpus: `training/corpus_sources/*.json`
- Canonical combined corpus: `data/train/datasets/corpus.jsonl`
- Canonical train split: `data/train/datasets/train.jsonl`
- Canonical validation split: `data/train/datasets/val.jsonl`
- Canonical holdout split: `data/train/datasets/holdout.jsonl`
- Fixtures: `data/train/datasets/fixtures.jsonl`
- Metadata: `data/train/datasets/metadata.json`

## Metadata

- `schema`: active schema id.
- `rows`: total rows written.
- `split_rows`: counts by split.
- `unique_rows`: normalized unique row count.
- `duplicate_rows`: normalized duplicate row count.
- `sources`: ordered source list with license and provenance details.

## Validation

- Source validation requires each JSON source entry to contain `tags` and
  object-shaped `content`.
- Validation requires at least one row in every emitted split file.
- Each row must contain valid `messages`, `tags`, and `meta`.
- Validation must fail on missing split labels or missing provenance fields.
- Validation proves shape only, not quality.

## Split Policy

- Split is row-based, not token-stream-based.
- Training uses `train.jsonl`.
- Validation loss uses `val.jsonl`.
- Behavioral evaluation uses `holdout.jsonl`.
