# Preference Optimization

Owner: `docs/architecture/training/evaluation/preference.md`.
State: target preference optimization contract.

## Goal

Improve the supervised scratch checkpoint with lightweight preference training
before attempting rollout-heavy reinforcement learning.

## Contract

- SimPO is the first preference optimization method because it avoids a
  reference model on 8 GiB hardware.
- DPO remains a comparison baseline only.
- GRPO/RLVR is a later pass after supervised and DPO evals are meaningful.
- Preference data is generated from behavioral eval cases and sampled model
  failures.
- Each preference row contains prompt messages, chosen assistant action, rejected
  assistant action, source, and reason.
- Chosen actions must be valid XML actions.
- Rejected actions may be invalid JSON, wrong tool selection, wrong final
  answer, or unsafe path usage.

## Command Status

Preference optimization is a design target. The active native training CLI does
not yet implement preference preparation, SimPO, or DPO commands.

## Artifacts

- Preference pairs: `data/train/preferences/pairs.jsonl`
- SimPO summary: `data/train/checkpoints/simpo-summary.json`
- SimPO checkpoint: `data/train/checkpoints/simpo/`
- DPO comparison summary: `data/train/checkpoints/dpo-summary.json`

## Acceptance

- Run SFT behavioral eval before preference training.
- Run behavioral eval again after SimPO.
- Accept SimPO only if pass rate improves or stays equal while reducing invalid
  XML and wrong-tool failures.
- Preference summaries default to `accepted=false` until comparison is recorded.
