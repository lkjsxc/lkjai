# Agent Competency Gate Contract

## Canonical Rule

An accepted model must pass fixed eval and beat the previous shipped model on
raw holdout behavior.

## Thresholds

- JSON validity on holdout: `>= 0.95`
- Holdout read/search/history/preview success: `>= 0.60`
- Holdout confirmation-planning success: `>= 0.50`

## Enforcement

- `TRAIN_ENFORCE_COMPETENCY=1` fails the pipeline when behavioral thresholds are
  missed.
- Quick smoke workflows may disable enforcement.
- Exact-match supervised lookup does not count toward competency.
