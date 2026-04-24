# Training Iteration Log

## Goal

Keep model-improvement claims tied to real artifacts, commands, and raw
behavioral reports.

## Baseline

- Artifact root: `data/models/lkjai-scratch-60m/`.
- Training summary: `data/train/checkpoints/training-summary.json`.
- Current parameter count: `55,866,240`.
- Current behavioral report: `data/train/runs/behavioral-eval.json`.
- Current pass rate: `0.235` from `47/200` cases.
- Current issue: malformed or prompt-copy generations were wrapped into valid
  fallback final actions, inflating XML validity.
- Current artifacts were trained on disallowed LLM-authored corpus content and
  are invalid for acceptance after the provenance policy change.
- New baseline target: 60,000 active rows with ≥57,000 unique and ≤1% duplicates.

## Iteration Command

```bash
TRAIN_PRESET=agent \
TRAIN_BEHAVIORAL_THRESHOLD=0.35 \
docker compose --profile train up --build --abort-on-container-exit train
```

## Acceptance Record

Each accepted run records:

- command and environment overrides,
- checkpoint source,
- fixed eval pass rate,
- behavioral pass rate,
- XML validity without fallback wrapping,
- direct-answer, tool-call, confirmation, safety, and agentic bucket rates,
- manual inference probes.

## Manual Probe Set

- `Say hello.`
- `List files in the workspace.`
- `Remember that I prefer concise plans.`
- `Search kjxlkj resources for release notes.`
- `Create a kjxlkj note with body "# Draft".`
- `Plan how to fix a failing test, then run the fix.`
- `Search docs for the deployment contract, then summarize it.`

## Update Rule

- Raise `TRAIN_BEHAVIORAL_THRESHOLD` after an accepted improvement.
- Do not lower the threshold to accept a weaker run.
- Do not record a run as improved unless raw generated actions beat the previous
  best pass rate.
