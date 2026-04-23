# Training Dataset

## Goal

Produce a valid, sizable dataset for scratch LM and agent behavior training.

## Contract

- The dataset is a JSONL file where each row contains `messages` and optional
  `tags`.
- `messages` follows the chat format: `role`, `content`, optional `name`.
- Roles: `system`, `user`, `assistant`, `tool`.
- The dataset must contain at least 4,000 rows for the `agent` preset.
- The dataset must cover all tool trajectories used by the agent.
- The dataset metadata must record source, license, row count, split, and schema
  version for each generated or imported source.

## Generation

- `prepare-fixtures` creates a minimal deterministic set for smoke checks.
- `prepare-corpus` generates synthetic trajectories for real training.
- Synthetic trajectories include: user request, assistant tool call, tool result,
  assistant final answer.
- Trajectories cover every tool: `shell.exec`, `web.fetch`, `fs.read`,
  `fs.write`, `fs.list`, `memory.search`, `memory.write`.
- kjxlkj trajectories cover search, read, summarize, create-note, update-note,
  and privacy-preserving organization tasks.
- Public instruction rows are optional, permissive-license, and marked with
  source tags.
- Direct-answer rows remain explicit so the model does not learn to answer every
  non-tool prompt from docs snippets.

## Scratch Formatting

- Dataset rows stay as structured messages.
- Tokenization uses the project scratch chat serializer.
- No upstream pretrained tokenizer template is allowed in the default path.

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

Expected: row count >= 4,000 for agent training.
