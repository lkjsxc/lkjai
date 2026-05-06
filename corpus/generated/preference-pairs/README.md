# Preference Pairs

## Purpose

This directory stores pairwise preference data for later SimPO or DPO work.
Rows here are never consumed as active SFT targets.

## Row Contract

Each row contains:

- `prompt_messages`
- `chosen_action`
- `rejected_action`
- `winner`
- `reason`
- `source_ref`
- `failure_type`

Chosen actions must be valid XML actions. Rejected actions may demonstrate
wrong tools, wrong fields, missing confirmation arguments, unsafe paths,
over-tooling, repeated failed actions, or fake tool results.

## Layout

```text
manifest.json
pairs/pairs-*.jsonl
```
