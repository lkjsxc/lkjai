# Corpus Fixtures

## Purpose

Fixture files describe grounded source snippets used by generated SFT and
preference rows.

## Contents

- [repo-grounding-v1.json](repo-grounding-v1.json): first grounding set derived
  from `lkjai`, `kjxlkj`, and `lkjmcsmp` docs.

## Rules

- Fixtures must reference real docs or runtime contracts.
- Fixtures must be small enough for LLM agents to inspect directly.
- Generated rows must cite fixture ids through `meta.fixture_id`.
