# Corpus Contract

## Source

- Default dataset: `HuggingFaceFW/fineweb-edu`.
- Default split: `train`.
- Default language focus: English educational text.
- Dataset use must preserve upstream license metadata in `data/corpus/`.

## Token Budgets

- Full-training default: approximately 3B tokens.
- Medium pipeline check: 50M tokens.
- Routine verification: tiny deterministic sample.

## Storage

- Raw or streamed corpus metadata lives under `data/corpus/raw`.
- Tokenized shards live under `data/corpus/tokenized`.
- Tokenizer artifacts live under `data/tokenizers`.
- Corpus data is never committed.
