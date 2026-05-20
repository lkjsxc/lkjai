# Kimi SFT Corpus

Owner: `ops/corpus/kimi_sft/README.md`.
State: operational.

## Purpose

This directory owns corpus-only commands for the Kimi-generated first-party SFT
dataset. Commands never write directly to active training data.

## Commands

- `generate`: write candidate JSONL shards under
  `data/corpus/quarantine/kimi-sft-60m`.
- `validate`: check quarantine or promoted shards for JSONL structure, one
  assistant XML action target, allowed tools, metadata, and split isolation.
- `promote`: copy only passing shards into
  `data/corpus/generated/kimi-sft-60m`.
- `report`: summarize row counts and validation status.
