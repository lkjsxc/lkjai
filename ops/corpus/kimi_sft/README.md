# Kimi SFT Corpus

Owner: `ops/corpus/kimi_sft/README.md`.
State: operational.

## Purpose

This directory owns corpus-only commands for the Kimi-generated first-party SFT
dataset. Commands never write directly to active training data.

## Commands

- `generate`: write candidate JSONL shards under
  `data/corpus/quarantine/kimi-sft-60m`. It reads
  `configs/corpus/kimi_sft_60m.yaml` by default, authenticates with
  `KIMI_API_KEY_FILE` or `KIMI_API_KEY`, calls the configured Kimi provider,
  skips already-valid shards, obeys `runs/kimi_corpus/STOP`, and writes
  `generation-report.json` without secret values. The default provider is
  `kimi-cli`, which delegates model calls through the native
  `lkjai-native-kimi-cli-runner` worker pool and the official `kimi` CLI.
  `KIMI_API_BASE_URL`, `KIMI_API_MODEL`, and `KIMI_USER_AGENT` can override the
  config for endpoint testing.
- `validate`: check quarantine or promoted shards for JSONL structure, one
  assistant XML action target, allowed tools, metadata, and split isolation.
- `promote`: copy only passing shards into
  `data/corpus/generated/kimi-sft-60m`.
- `report`: summarize row counts and validation status.

Generation stops with a report instead of discarding valid quarantine rows when
quota is exhausted. Authentication and access-termination failures remain hard
failures because no trusted Kimi output can be produced.
