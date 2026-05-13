# Scaling Laws

Owner: `docs/research/scaling-laws.md`.
State: research guidance.

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
- Chinchilla target: `~1.16B tokens`

## Practical Budget

Current public pretrain materialization is much larger than the old Kimi SFT
pack, but still short of compute-optimal scale:

- Public train tokenizer tokens: about `463M`
- Tokens per parameter at 40M: about `11.7`
- Chinchilla gap at 40M: about `41.5% shortfall`

The older Kimi SFT pack was about `26M` train tokens. It is not the current
public-pretrain budget and must not be used as the scale baseline.

This gap is expected and acceptable for the default path:

- The target hardware is a single RTX 3070 8GB.
- We optimize for trusted provenance, task diversity, and format alignment, not
  raw token volume.
- The 40M preset is the active compromise between capacity and RTX 3070 memory.
- Scaling beyond the current public-pretrain pack and rebuilding reviewed SFT
  data remain necessary before `scratch-60m` becomes a serious default
  candidate.

## SmolLM2 Guidance

SmolLM2 (Allal et al., 2025) shows that small models benefit more from
high-quality, deduplicated, task-diverse data than from scaling tokens alone.

Key takeaways for lkjai:

- Curate before scaling.
- Evaluate by task bucket, not aggregate loss.
- Match training serialization to inference prompts exactly.

## Token Budget Metadata

Native training reports record these fields:

- `train_tokens`: total tokenizer tokens on the train split
- `parameter_count`: model parameter count
- `tokens_per_parameter`: `train_tokens / parameter_count`
- `chinchilla_gap`: relative shortfall from 20 tokens/parameter

## References

- Chinchilla: <https://arxiv.org/abs/2203.15556>
- SmolLM2: <https://arxiv.org/abs/2502.02737>
