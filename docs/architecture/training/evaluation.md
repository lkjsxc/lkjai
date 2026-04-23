# Training Evaluation

## Goal

Measure whether training produced usable scratch artifacts.

## Contract

- Fixed eval runs after every training pipeline.
- Fixed eval checks artifact existence, dataset coverage, checkpoint structure,
  tokenizer structure, and loss metrics.
- Behavioral eval checks whether generated responses are valid and useful.
- Competency gate uses behavioral eval pass rate.

## Fixed Eval Checks

1. `fixtures-exist`: dataset file exists.
2. `dataset-metadata-exists`: metadata JSON exists.
3. `training-summary-exists`: summary JSON with metrics exists.
4. `tokenizer-manifest-exists`: tokenizer manifest JSON exists.
5. `checkpoint-manifest-exists`: scratch checkpoint manifest JSON exists.
6. `export-manifest-exists`: export manifest exists.
7. `tool-trajectory-present`: dataset contains tool call examples.
8. `memory-case-present`: dataset contains memory write examples.
9. `checkpoint-has-weights`: checkpoint directory contains model weights and
   config.
10. `summary-has-loss`: training summary contains non-empty metrics.

## Competency Gate

- Threshold default: `0.80`.
- Enforcement: `TRAIN_ENFORCE_COMPETENCY=1` fails the pipeline when below
  threshold.
- Artifact pass rate: `passed / total` fixed checks.
- Behavioral pass rate: prompt cases passed by actual model inference.

## Report Schema

```json
{
  "threshold": 0.8,
  "pass_rate": 0.9,
  "passed": 9,
  "total": 10,
  "cases": [
    {"id": "fixtures-exist", "passed": true, "detail": "..."}
  ]
}
```

## Verification

```bash
python -m lkjai_train.cli fixed-eval
python -m lkjai_train.cli behavioral-eval
cat data/train/runs/fixed-eval.json | jq .pass_rate
cat data/train/runs/behavioral-eval.json | jq .pass_rate
```
