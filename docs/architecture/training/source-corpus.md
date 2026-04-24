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

## Active Source Files

- `public.json`: opt-in permissive public dataset allowlist with license,
  revision, source URL, local file, and row limit.

## Quarantined Source Files

- `agentic_plan.json`
- `agentic_tools.json`
- `agentic_revision.json`
- `docs_grounding.json`
- `general.json`
- `kjxlkj.json`

These packs are not active training data because they are LLM-authored corpus
content. They may remain as reference material, but `prepare-corpus` must not
consume them by default.

## Rationale

- JSON arrays are easier for LLM agents to inspect, diff, and extend than Python
  constant blocks.
- JSONL remains the generated training artifact because it streams cleanly and
  matches existing tokenizer/training code.
- Tags make future filtering, provenance checks, and corpus balancing explicit.
