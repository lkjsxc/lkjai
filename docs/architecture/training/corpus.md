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
- `prepare-corpus` generates synthetic trajectories covering all tools plus
  docs-grounded answers and kjxlkj organization examples.
- Each trajectory contains a user request, an assistant tool call, a tool result,
  and a final assistant answer.
- Corpus size is configurable via `TRAIN_CORPUS_SIZE`.
- Default corpus size for `agent` preset: 4,000.
- Default mix: 40% docs-grounding, 25% tool and memory trajectories, 20% kjxlkj
  organization trajectories, and 15% vetted public instruction rows.
- Public rows are skipped when unavailable unless explicitly required.

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

## Prompt And Action Format

- Runtime prompts use paired section tags such as `<run>`, `<summary>`,
  `<memories>`, and `<events>` for segmentation.
- Assistant outputs remain strict JSON actions, not XML, because tool execution
  needs typed validation.
- Training and inference must use the same serializer.

## Verification

```bash
python -m lkjai_train.cli prepare-corpus
python -m lkjai_train.cli validate-dataset
jq -c 'select(.tags | contains(["tool_trajectory"]))' data/train/datasets/corpus.jsonl | wc -l
```

Expected: tool_trajectory count >= 1,000 for agent training.
