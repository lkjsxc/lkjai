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
  fallback final actions, inflating JSON validity.

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
- JSON validity without fallback wrapping,
- direct-answer, tool-call, confirmation, and safety bucket rates,
- manual inference probes.

## Manual Probe Set

- `Say hello.`
- `List files in the workspace.`
- `Remember that I prefer concise plans.`
- `Search kjxlkj resources for release notes.`
- `Create a kjxlkj note with body "# Draft".`

## Update Rule

- Raise `TRAIN_BEHAVIORAL_THRESHOLD` after an accepted improvement.
- Do not lower the threshold to accept a weaker run.
- Do not record a run as improved unless raw generated actions beat the previous
  best pass rate.
