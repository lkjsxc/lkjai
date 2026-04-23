# Corpus Sources

This directory contains editable corpus source entries.

## Contract

- Each `*.json` file is one JSON array.
- Each array item has `tags` and `content`.
- `tags` is a non-empty string array.
- `content` is a JSON object.
- Python code may expand these entries, but large authored lists belong here.

## Files

- `general.json`: reasoning, safety, local tool, and source metadata entries.
- `kjxlkj.json`: resource API terms, refs, note bodies, visibility rules, and
  tool-control values.

## Check

Run:

```sh
PYTHONPATH=training python3 -m lkjai_train.cli validate-sources
```
