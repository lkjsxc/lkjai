# Reasoning Models

## Goal

Summarize applicable lessons from recent reasoning-budget and RL work for our
from-scratch agent pipeline.

## Visible Planning Over Hidden CoT

DeepSeek-R1 (Guo et al., 2025) and Qwen3 (Qwen Team, 2025) both show that
reasoning models can learn to plan before acting. For small scratch models:

- Hidden chain-of-thought fields are harder to supervise.
- Visible `plan` actions in the message stream are easier to evaluate and
  correct.
- The plan becomes part of the supervised target, not an auxiliary loss.

## Trajectory Correctness

Reasoning models are often trained with RL on outcome correctness. Our pipeline
approximates this with:

- Behavioral eval on raw holdout generations.
- DPO on preference pairs derived from holdout rows.
- Fixed eval gates that require exact action shapes.

## Training Step Scaling

Raising default steps from 3,000 to 12,000 is a lightweight proxy for
reasoning-model training:

- More steps let the model overfit less and generalize format patterns.
- Stable LR with cosine decay keeps late-step updates useful.
- The corpus is small enough that 12k steps ≈ 2.7 epochs, not excessive
  repetition.

## What We Do Not Adopt

- Large-scale RL with process reward models (too expensive for scratch-60m).
- Distillation from pretrained reasoning models (out of scope for default path).
- Dynamic reasoning budgets (fixed max tokens keep serving predictable).

## References

- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- Qwen3: <https://arxiv.org/abs/2505.09388>
