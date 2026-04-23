# Source Corpus

## Goal

Keep authored training content in LLM-readable JSON files instead of hiding it
inside Python constants.

## Location

- Source directory: `training/corpus_sources/`
- Source files: `*.json`
- Python loader: `training/lkjai_train/corpus_source.py`
- Generated dataset: `data/train/datasets/corpus.jsonl`

## File Shape

Each source file is one JSON array. Each element has tags and content:

```json
[
  {
    "tags": ["general_topic", "language:en"],
    "content": {
      "topic": "debugging",
      "practice": "reproduce, isolate, fix, verify",
      "failure": "premature guesses"
    }
  }
]
```

## Rules

- `tags` is a non-empty array of strings.
- `content` is a JSON object.
- Entries are human-reviewable and LLM-editable.
- Generation code may combine entries, but must not own large authored lists.
- JSON source files are commercial-safe project content unless a license entry
  states otherwise.
- Public-source entries must include license and source metadata.

## Current Source Files

- `general.json`: reasoning topics, prompt variants, safety boundaries, local
  tool scenarios, and source metadata.
- `kjxlkj.json`: resource terms, refs, note bodies, update bodies, preview
  bodies, visibility rules, search kinds, and history windows.

## Rationale

- JSON arrays are easier for LLM agents to inspect, diff, and extend than Python
  constant blocks.
- JSONL remains the generated training artifact because it streams cleanly and
  matches existing tokenizer/training code.
- Tags make future filtering, provenance checks, and corpus balancing explicit.
