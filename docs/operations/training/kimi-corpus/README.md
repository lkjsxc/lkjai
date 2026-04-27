# Kimi Corpus

## Purpose

This directory is the canonical runbook for generating the English-only
Kimi-authored synthetic corpus.

The active objective is `sft`: XML-action chat rows for assistant-masked
supervised training. Public Cosmopedia text supplies the causal-LM pretraining
side.

The refreshed long-run target is `60000000` generated SFT tokens. The historical
committed corpus lives under `corpus/generated/kimi-full-v1/` until a refreshed
`corpus/generated/kimi-sft-60m-v1/` corpus is validated.

## Read Order

1. [schema.md](schema.md): row formats, paths, and metadata.
2. [quality.md](quality.md): validation and scoring gates.
3. [workflow.md](workflow.md): sample-first and prompt-refinement flow.
4. [commands.md](commands.md): exact local commands.
5. [long-run.md](long-run.md): 60M-token background operation.

## Rules

- Use Kimi CLI non-interactively.
- Keep raw Kimi logs under `runs/kimi_corpus/logs/`.
- Generate English rows only.
- Use public corpus projects as quality references, not copied text.
- Do not paste full generated documents or ordinary CLI UI text into agent
  conversation.
- Validate before committing generated shards.
- Commit generated validated shards under `corpus/generated/kimi-sft-60m-v1/`.
- Keep generator staging outputs separated by objective. The committed active
  full corpus is normalized into `train`, `val`, and `holdout` splits.
