# Scaling Laws

## Goal

Track the practical gap between our from-scratch training budget and
compute-optimal token targets.

## Chinchilla Accounting

Hoffmann et al. (2022) defines compute-optimal training as roughly
20 tokens per parameter:

- `compute_optimal_tokens ≈ parameters × 20`

For our scratch-60m preset:

- Parameters: `~55,866,240`
- Chinchilla target: `~1.1T tokens`

## Practical Budget

At 6,000 active docs-derived corpus rows and ~150 tokens per row:

- Train tokens: `~900,000`
- Tokens per parameter: `~0.016`
- Chinchilla gap: `~1.1T - 0.9M ≈ 99.9% shortfall`

This gap is expected and acceptable for the default path:

- The target hardware is a single RTX 3070 8GB.
- We optimize for trusted provenance, task diversity, and format alignment, not
  raw token volume.
- Longer training steps (12,000) improve utilization of the available corpus.

## SmolLM2 Guidance

SmolLM2 (Allal et al., 2025) shows that small models benefit more from
high-quality, deduplicated, task-diverse data than from scaling tokens alone.

Key takeaways for lkjai:

- Curate before scaling.
- Evaluate by task bucket, not aggregate loss.
- Match training serialization to inference prompts exactly.

## Token Budget Metadata

`scratch_train.py` records these fields in `training-summary.json`:

- `train_tokens`: total tokenizer tokens on the train split
- `parameter_count`: model parameter count
- `tokens_per_parameter`: `train_tokens / parameter_count`
- `chinchilla_gap`: relative shortfall from 20 tokens/parameter

## References

- Chinchilla: <https://arxiv.org/abs/2203.15556>
- SmolLM2: <https://arxiv.org/abs/2502.02737>
