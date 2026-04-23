# Reinforcement Learning Research

## Current Policy

- Use DPO-style preference optimization before rollout-based RL.
- Defer GRPO/RLVR until supervised and DPO checkpoints have useful behavioral
  eval signal.
- Do not let RL replace dataset quality, prompt format, or sandbox fixes.

## DPO

- DPO trains from chosen/rejected response pairs.
- It is simpler than PPO-style RLHF for this local project.
- It fits the current eval loop because failures can become rejected examples.
- Reference: <https://arxiv.org/abs/2305.18290>

## GRPO And RLVR

- GRPO reduces value-model overhead compared with PPO-style approaches.
- It is useful inspiration for future rule-reward rollouts.
- It is not the first implementation because this project needs reliable
  supervised behavior and reward cases first.
- References: <https://arxiv.org/abs/2402.03300> and
  <https://arxiv.org/abs/2501.12948>

## DAPO

- DAPO is useful large-scale RL process research.
- It is too heavy for the immediate RTX 3070 implementation.
- Reference: <https://arxiv.org/abs/2503.14476>
