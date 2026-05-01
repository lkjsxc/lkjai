# Kimi Corpus Generation

## Goal

Generate optional Kimi-authored XML action data for the scratch agent, with
enough everyday conversation coverage to make basic chat usable after
pretraining.

For the current Kimi API pipeline, use [kimi-corpus/README.md](kimi-corpus/README.md).
This document is SFT/tool-data background; the active public pretraining path is
Cosmopedia `text` only.

## Target

- Pilot SFT tokenizer tokens: `1000000`.
- Full SFT tokenizer tokens: `60000000`.
- Chunk size: about `1000` JSONL rows.
- Pilot staging location: `data/kimi_synthetic/pilot-v2/`.
- Committed location: `corpus/generated/kimi-sft-60m-v2/`.
- Runtime staging location: `data/kimi_synthetic/` for optional Kimi rows.

## Mix

- Direct finish and preference handling: `15%`.
- Read-only retrieval and grounded answering: `35%`.
- Mutation with confirmation: `25%`.
- Failure, safety, and recovery: `25%`.

## Quality Gates

- XML validity rate: `>= 0.995`.
- Completed traces end with `agent.finish`: `1.0`.
- Confirmation-planning traces end with `agent.request_confirmation`.
- Runtime contract parse rate: `1.0`.
- Replayable confirmation rate: `1.0`.
- Duplicate rate: `<= 0.01`.
- Generic final-answer rate: `<= 0.005`.
- `Completed task for ...` rate in everyday-chat rows: `0`.
- Each split has everyday conversation rows.
- Everyday-chat holdout pass rate is reported separately.
- Non-final chunks have roughly `1000` rows.
- Every active row has `kimi-generated` provenance.

## Kimi API Requirements

- Base URL: `https://api.moonshot.ai/v1`.
- Default model: `kimi-k2.6`.
- Use JSON object output with top-level `rows`.
- Use `thinking={"type":"disabled"}` for canonical rows.
- Use `/tokenizers/estimate-token-count` before large shard requests.
- Use all discovered API keys in parallel and redact them everywhere.
- Treat API error types as retry, quota, auth, or quarantine signals.
