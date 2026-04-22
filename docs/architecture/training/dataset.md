# Training Dataset

## Goal

Produce a valid, sizable dataset for agent behavior tuning.

## Contract

- The dataset is a JSONL file where each row contains `messages` and optional
  `tags`.
- `messages` follows the chat format: `role`, `content`, optional `name`.
- Roles: `system`, `user`, `assistant`, `tool`.
- The dataset must contain at least 100 rows for the `agent` preset.
- The dataset must cover all tool trajectories used by the agent.

## Generation

- `prepare-fixtures` creates a minimal deterministic set for smoke checks.
- `prepare-corpus` generates synthetic trajectories for real training.
- Synthetic trajectories include: user request, assistant tool call, tool result,
  assistant final answer.
- Trajectories cover every tool: `shell.exec`, `web.fetch`, `fs.read`,
  `fs.write`, `fs.list`, `memory.search`, `memory.write`.

## Chat Template

- Tokenization must use `tokenizer.apply_chat_template(messages, tokenize=False)`
  instead of naive string concatenation.
- This ensures the base model's expected formatting is preserved.

## Validation

- Every row must have a non-empty `messages` list.
- Every message must have a valid `role` and string `content`.
- Validation runs before GPU training starts.

## Verification

```bash
python -m lkjai_train.cli prepare-corpus
python -m lkjai_train.cli validate-dataset
wc -l data/train/datasets/corpus.jsonl
```

Expected: row count >= 100 for agent training.
