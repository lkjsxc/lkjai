# Tuning Data Research

## Default Data Shape

- Use standalone English `pretrain` JSONL for the active 500M-token corpus.
- Preserve OpenAI-style `messages` JSONL for later SFT and tool trajectories.
- Public rows must record source URL, license, revision, and row count.
- Default public imports avoid share-alike and attribution-only licenses.

## Public Candidates

- Cosmopedia: Apache-2.0 English educational pretraining source.
- OpenAssistant OASST1: permissive Apache-2.0 instruction data candidate for
  later SFT.
- FineWeb, Dolma, and OpenWebMath are high-quality references but excluded from
  active data under the current Apache/MIT/BSD-only policy.

## Verification Dataset

- Verification uses tiny local fixtures.
- Verification does not download large datasets.
- Fixture cases must cover chat, tool use, memory, and invalid action handling.

## Metadata Rule

- Dataset name, source, license, split, row count, and schema version are written
  beside prepared dataset artifacts.
