# Preference Optimization

## Goal

Improve the supervised scratch checkpoint with lightweight preference training
before attempting rollout-heavy reinforcement learning.

## Contract

- DPO is the first preference optimization method.
- GRPO/RLVR is a future phase after supervised and DPO evals are meaningful.
- Preference data is generated from behavioral eval cases and sampled model
  failures.
- Each preference row contains prompt messages, chosen assistant action, rejected
  assistant action, source, and reason.
- Chosen actions must be valid XML actions.
- Rejected actions may be invalid JSON, wrong tool selection, wrong final
  answer, or unsafe path usage.

## Commands

- `python -m lkjai_train.cli prepare-preferences`
- `python -m lkjai_train.cli train-dpo`

## Artifacts

- Preference pairs: `data/train/preferences/pairs.jsonl`
- DPO summary: `data/train/checkpoints/dpo-summary.json`
- DPO checkpoint: `data/train/checkpoints/dpo/`

## Acceptance

- Run SFT behavioral eval before DPO.
- Run behavioral eval again after DPO.
- Accept DPO only if pass rate improves or stays equal while reducing invalid
  JSON and wrong-tool failures.
- DPO summaries default to `accepted=false` until that comparison is recorded.
