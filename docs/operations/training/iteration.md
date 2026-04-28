# Training Iteration Log

## Goal

Keep model-improvement claims tied to real artifacts, commands, and raw
behavioral reports.

## Baseline

- Artifact root: `data/models/lkjai-scratch-40m/`.
- Training summary: `data/train/checkpoints/training-summary.json`.
- Active default parameter target: about `40M`.
- Current behavioral report: `data/train/runs/behavioral-eval.json`.
- Current pass rate: `0.0` from `0/200` cases.
- Current issue: malformed or prompt-copy generations were wrapped into valid
  fallback final actions, inflating XML validity.
- Current artifacts were trained on disallowed LLM-authored corpus content and
  are invalid for acceptance after the provenance policy change.
- New baseline target: `500000000` public English pretraining tokens plus
  `60000000` first-party XML-action SFT tokens.

Materialize the ignored corpus after downloading Cosmopedia to
`data/raw/cosmopedia/`:

```bash
docker compose --profile corpus run --rm corpus download-public-pretrain
docker compose --profile corpus run --rm corpus prepare-public-pretrain
```

## Iteration Command

Fresh 500M-target run, started from an empty data directory:

```bash
docker compose --profile train run --rm \
  -e DATA_DIR=/app/data/train-full-500m-from-scratch-v2 \
  -e TRAIN_RESUME=never \
  -e TRAIN_INIT_CHECKPOINT= \
  -e TRAIN_CORPUS_DIR=/app/data/public-corpus \
  -e TRAIN_PUBLIC_DATA_DIR=/app/data/raw/cosmopedia \
  -e TRAIN_PUBLIC_PRETRAIN_TOKENS=500000000 \
  -e TRAIN_CONFIG=/workspace/configs/training/scratch_40m_12h.json \
  -e TRAIN_PRESET=agent \
  train train
```

Stopped first attempt, because it used the previous `440000000` public target:

- Step `1`: loss `9.1082`, `8192` input tokens seen.
- Step `3000`: loss `6.5823`, `24576000` input tokens seen.
- Step `6000`: loss `6.9035`, `49152000` input tokens seen.

Corrected 500M public run:

- Data directory: `data/train-full-500m-from-scratch-v2/`.
- Public train tokenizer tokens: `463087933`.
- Step `1`: loss `9.10818`, `8192` input tokens seen.
- Step `3000`: loss `6.9528`, `24576000` input tokens seen.
- Step `6000`: loss `7.1408`, `49152000` input tokens seen.
- Step `9000`: loss `7.3577`, `73728000` input tokens seen.
- Step `12000`: loss `7.4170`, `98304000` input tokens seen.

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
