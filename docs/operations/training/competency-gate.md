# Competency Gate Contract

## Canonical Threshold

- Competency is defined as fixed-eval pass rate `>= 0.80`.
- Fixed-eval results are written to `data/train/runs/fixed-eval.json`.

## Fixed-Eval Schema

- `threshold`: configured threshold.
- `pass_rate`: `passed / total`.
- `passed`: number of passing checks.
- `total`: total checks.
- `cases`: ordered list of check records:
  - `id`: stable case identifier.
  - `passed`: boolean.
  - `detail`: concise machine-readable reason.

## Acceptance Rule

1. If `pass_rate >= threshold`, the run is competency-accepted.
2. If `pass_rate < threshold`, continue training with another long-run cycle.
3. Keep artifact history in `data/train/runs/` for audit and comparison.

## Enforcement

- `TRAIN_FIXED_EVAL_THRESHOLD` configures the threshold (default `0.80`).
- `TRAIN_ENFORCE_COMPETENCY=1` fails the training pipeline when below threshold.
- Quick/smoke workflows may disable enforcement for fast local checks.
