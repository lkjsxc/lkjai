# Dataset Contract

## Format

- Training data uses JSONL.
- Each row contains OpenAI-style `messages`.
- Tool trajectory rows include assistant JSON action messages and tool results.
- Memory rows include retrieved memory and expected final behavior.

## Sources

- In-repo fixtures cover verification only.
- Larger tuning datasets live under `data/train/datasets/`.
- External datasets must preserve source and license metadata.

## Required Splits

- `instruction`: compact multi-turn instruction behavior.
- `tool_trajectory`: plan, tool call, observation, final answer.
- `memory`: recall, summary use, and durable memory writes.
- `eval`: deterministic fixed cases.

## Non-Goal

- Do not use a 3B-token pretraining corpus as the default path.
