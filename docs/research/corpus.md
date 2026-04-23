# Tuning Data Research

## Default Data Shape

- Use OpenAI-style `messages` JSONL.
- Preserve tool-call actions and tool results as supervised trajectory data.
- Include memory retrieval and memory write examples.
- Include kjxlkj resource organization and note maintenance examples.
- Optional public rows must record source URL, license, revision, and row count.
- Default public imports avoid share-alike licenses.

## Public Candidates

- OpenAssistant OASST1: permissive Apache-2.0 instruction data candidate.
- ToolBench: permissive Apache-2.0 tool-use data candidate.
- Dolly is excluded by default because CC BY-SA needs an explicit future
  share-alike decision.

## Verification Dataset

- Verification uses tiny local fixtures.
- Verification does not download large datasets.
- Fixture cases must cover chat, tool use, memory, and invalid action handling.

## Metadata Rule

- Dataset name, source, license, split, row count, and schema version are written
  beside prepared dataset artifacts.
