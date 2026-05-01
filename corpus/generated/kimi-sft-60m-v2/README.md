# Kimi SFT 60M V2

## Purpose

This directory is the accepted destination for the rebuilt first-party
XML-action SFT corpus.

## Acceptance

- Target tokenizer tokens: `60000000`.
- Schema: `lkjai-agent-jsonl-v3`.
- Required splits: `train`, `val`, and `holdout`.
- Every assistant message is one XML `<action>`.
- Completed traces end with `agent.finish`.
- Confirmation traces may end with `agent.request_confirmation` only when the
  stored pending mutation is replayable.
- Every row passes runtime action validation.
- Every row references a committed fixture or whitelisted repo document.
- Preference comparisons are excluded.

## Layout

- [manifest.json](manifest.json): corpus status and split counts.
- [validation-report.json](validation-report.json): latest committed
  validation summary.
- `train/train-*.jsonl`: promoted train shards.
- `val/val-*.jsonl`: promoted validation shards.
- `holdout/holdout-*.jsonl`: promoted holdout shards.

## Current State

The committed rows are a seed slice for validator and fixture testing. They are
not enough for training acceptance.

Generated staging outputs stay under ignored `data/kimi_synthetic/` until they
pass all gates.
