# Training Corpus

## Goal

Define the dataset that teaches the agent multi-turn tool use.

## Schema

Each JSONL row:

```json
{
  "messages": [
    {"role": "user", "content": "List this directory."},
    {"role": "assistant", "content": "{\"kind\":\"tool_call\",\"thought\":\"inspect\",\"tool\":\"fs.list\",\"args\":{\"path\":\".\"}}"},
    {"role": "tool", "name": "fs.list", "content": "README.md\nsrc"},
    {"role": "assistant", "content": "{\"kind\":\"final\",\"thought\":\"done\",\"content\":\"README.md and src\"}"}
  ],
  "tags": ["tool_trajectory", "fs.list"]
}
```

## Generation Strategy

- `prepare-fixtures` writes a minimal deterministic set (2 rows) for smoke
  checks.
- `prepare-corpus` generates synthetic trajectories covering all tools.
- Each trajectory contains a user request, an assistant tool call, a tool result,
  and a final assistant answer.
- Corpus size is configurable via `TRAIN_CORPUS_SIZE`.
- Default corpus size for `agent` preset: 200.

## Tool Coverage

Every generated corpus must include examples for:

- `shell.exec`
- `web.fetch`
- `fs.read`
- `fs.write`
- `fs.list`
- `memory.search`
- `memory.write`

## Scratch Tokenization

- Train a byte-level BPE tokenizer from local corpus text.
- Do not use a pretrained tokenizer as the default.
- Do not pre-format messages as strings in the dataset.
- Keep the dataset as structured `messages` arrays.

## Verification

```bash
python -m lkjai_train.cli prepare-corpus
python -m lkjai_train.cli validate-dataset
jq -c 'select(.tags | contains(["tool_trajectory"]))' data/train/datasets/corpus.jsonl | wc -l
```

Expected: tool_trajectory count >= 50.
