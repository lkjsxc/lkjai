# Small Model Training Notes

## Policy

- Use recent small-model work as process guidance, not as pretrained weights.
- Keep the default runtime trained from random initialization.
- Import public data only when license and revision metadata are explicit.

## Data-Centric Lessons

- Small models need curated, deduplicated, task-balanced data more than sheer
  row count.
- Holdout splits must include tool selection, direct answers, confirmations,
  safety boundaries, docs grounding, and agentic multi-turn trajectories.
- Evaluation must report task buckets so a single aggregate pass rate does not
  hide tool or confirmation regressions.
- Use the 40M preset as the active RTX 3070 baseline. Move to 60M only after
  data volume and behavior justify it.

## Optimizer Lessons

- AdamW remains the default optimizer.
- Muon is an optional experiment after the baseline improves because it changes
  optimizer behavior and requires its own acceptance comparison.
- Any optimizer change must keep the same corpus, seed, and eval cases for an
  apples-to-apples run.

## Token Budget

- See [scaling-laws.md](scaling-laws.md) for Chinchilla-style accounting and
  the practical gap between our scratch budget and compute-optimal targets.

## References

- SmolLM2 data-centric small-language-model training:
  <https://arxiv.org/abs/2502.02737>
- Muon scalability study:
  <https://arxiv.org/abs/2502.16982>
- Grouped-query attention:
  <https://arxiv.org/abs/2305.13245>
- SimPO:
  <https://arxiv.org/abs/2405.14734>
- Direct Preference Optimization:
  <https://arxiv.org/abs/2305.18290>
