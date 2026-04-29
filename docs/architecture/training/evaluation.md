# Training Evaluation

## Goal

Define the checks that decide whether a model is better in the real runtime.

## Fixed Eval

- Fixed eval verifies artifacts, data integrity, dedup quality, and split shape.
- It writes `data/train/runs/fixed-eval.json`.
- It must fail when the corpus remains mostly repeated boilerplate.

## Behavioral Eval

- Behavioral eval runs against the exported model with raw generation only.
- It writes `data/train/runs/behavioral-eval.json`.
- It reads cases from the holdout split.
- There is no supervised action lookup in the accepted path.
- Invalid XML, wrong action tool, missing required fields, and wrong tool
  arguments fail the case.
- Eval reports must not wrap malformed generation into a valid fallback action.

## Current Baseline

- Baseline artifact: `data/train/runs/behavioral-eval.json`.
- Current accepted baseline: none.
- Latest documented raw repair runs remain `0/200` behavioral pass rate.
- Historical `0.235` pass rate was fallback-inflated and is not an acceptance
  baseline.

## Acceptance Metrics

- XML validity on holdout: `>= 0.95`
- Holdout read/search/history/preview success: `>= 0.60`
- Holdout create/update confirmation-planning success: `>= 0.50`
- Any accepted run must beat the previous shipped model on raw holdout pass
  rate.

## Failure Handling

- Fixed eval failure blocks acceptance immediately.
- Behavioral regression blocks export acceptance.
- Preference training is optional and must be rejected if it lowers raw holdout
  quality.
