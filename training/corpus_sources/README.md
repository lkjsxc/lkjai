# Corpus Sources

This directory contains editable corpus source entries.

## Contract

- Each `*.json` file is one JSON array.
- Each array item has `tags` and `content`.
- `tags` is a non-empty string array.
- `content` is a JSON object.
- Python code may expand these entries, but large authored lists belong here.

## Active Files

- `public.json`: public dataset candidates, legal-review entries, and any
  explicitly activated `public_dataset` rows with license, pinned revision, and
  local-file metadata.

## Quarantined Files

- `agentic_plan.json`
- `agentic_tools.json`
- `agentic_revision.json`
- `docs_grounding.json`
- `general.json`
- `kjxlkj.json`

These files contain LLM-authored corpus content and are not consumed by the
default training corpus.

## Check

Run:

```sh
PYTHONPATH=training python3 -m lkjai_train.cli validate-sources
```
