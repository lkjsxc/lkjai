# Scaling Laws

## Goal

Track the practical gap between our from-scratch training budget and
compute-optimal token targets.

## Chinchilla Accounting

Hoffmann et al. (2022) defines compute-optimal training as roughly
20 tokens per parameter:

- `compute_optimal_tokens ≈ parameters × 20`

For the active `scratch-40m` preset:

- Parameters: `~39,567,168`
- Chinchilla target: `~791M tokens`

For the long-term `scratch-60m` preset:

- Parameters: `~58M`
- Chinchilla target: `~1.1T tokens`

## Practical Budget

At the current committed Kimi corpus size:

- Train tokens: about `26M`
- Tokens per parameter at 40M: about `0.66`
- Chinchilla gap: about `96.7% shortfall`

This gap is expected and acceptable for the default path:

- The target hardware is a single RTX 3070 8GB.
- We optimize for trusted provenance, task diversity, and format alignment, not
  raw token volume.
- The 40M preset is the active compromise between capacity and RTX 3070 memory.
- Scaling toward 500M tokens remains necessary before `scratch-60m` becomes a
  serious default candidate.

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
