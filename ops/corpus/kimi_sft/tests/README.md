# Kimi SFT Tests

Owner: `ops/corpus/kimi_sft/tests/README.md`.
State: operational.

## Purpose

Unit tests for the corpus-only Kimi SFT commands.

## Files

- [test_cli.py](test_cli.py): validates API-key loading, row validation,
  promotion, pilot gating, and Kimi CLI runner command behavior.

## Rules

- Tests must not call the real Kimi API or official CLI.
- Secrets must stay in environment variables or temporary files only.
