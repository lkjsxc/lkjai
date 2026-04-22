# Tuning Data Research

## Default Data Shape

- Use OpenAI-style `messages` JSONL.
- Preserve tool-call actions and tool results as supervised trajectory data.
- Include memory retrieval and memory write examples.

## Verification Dataset

- Verification uses tiny local fixtures.
- Verification does not download large datasets.
- Fixture cases must cover chat, tool use, memory, and invalid action handling.

## Metadata Rule

- Dataset name, source, license, split, row count, and schema version are written
  beside prepared dataset artifacts.
