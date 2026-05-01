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

```text
manifest.json
validation-report.json
train/train-*.jsonl
val/val-*.jsonl
holdout/holdout-*.jsonl
```

Generated staging outputs stay under ignored `data/kimi_synthetic/` until they
pass all gates.
