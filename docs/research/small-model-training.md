# Small Model Training Notes

## Policy

- Use recent small-model work as process guidance, not as pretrained weights.
- Keep the default runtime trained from random initialization.
- Import public data only when license and revision metadata are explicit.

## Data-Centric Lessons

- Small models need curated, deduplicated, task-balanced data more than sheer
  row count.
- Holdout splits must include tool selection, direct answers, confirmations,
  safety boundaries, and docs grounding.
- Evaluation must report task buckets so a single aggregate pass rate does not
  hide tool or confirmation regressions.

## Optimizer Lessons

- AdamW remains the default optimizer.
- Muon is an optional experiment after the baseline improves because it changes
  optimizer behavior and requires its own acceptance comparison.
- Any optimizer change must keep the same corpus, seed, and eval cases for an
  apples-to-apples run.

## References

- SmolLM2 data-centric small-language-model training:
  <https://arxiv.org/abs/2502.02737>
- Muon scalability study:
  <https://arxiv.org/abs/2502.16982>
- Grouped-query attention:
  <https://arxiv.org/abs/2305.13245>
- Direct Preference Optimization:
  <https://arxiv.org/abs/2305.18290>
