# Corpus Contract

## Source

- Default dataset: `HuggingFaceFW/fineweb-edu`.
- Default split: `train`.
- Default language focus: English educational text.
- Dataset use must preserve upstream license metadata in `data/train/corpus/`.

## Token Budgets

- Full-training default: approximately 3B tokens.
- Medium pipeline check: 50M tokens.
- Routine verification: tiny deterministic sample.

## Storage

- Raw or streamed corpus metadata lives under `data/train/corpus/raw`.
- Tokenized shards live under `data/train/corpus/tokenized`.
- Tokenizer artifacts live under `data/train/tokenizers`.
- Corpus data is never committed.
