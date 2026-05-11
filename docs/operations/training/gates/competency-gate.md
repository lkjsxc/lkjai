# Agent Competency Gate Contract

## Canonical Rule

An accepted model must pass fixed eval and beat the previous shipped model or
current documented baseline on raw holdout behavior.

## Current Ladder

- Current accepted baseline: none. The current artifact reports
  `pass_rate=0.0` from `0/200` raw holdout cases.
- Historical `pass_rate=0.235` was inflated by fallback wrapping and is not an
  acceptance baseline.
- Next improvement gate: `TRAIN_BEHAVIORAL_THRESHOLD=0.35`.
- Raise the threshold after each accepted model run.
- Do not call the assistant competent until the thresholds below are met.

## Thresholds

- XML validity on holdout: `>= 0.95`
- Holdout read/search/history/preview success: `>= 0.60`
- Holdout confirmation-planning success: `>= 0.50`

## Enforcement

- `TRAIN_ENFORCE_COMPETENCY=1` fails the pipeline when behavioral thresholds are
  missed.
- Quick smoke workflows may disable enforcement.
- Exact-match supervised lookup does not count toward competency.
