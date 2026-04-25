# Kimi Corpus

## Purpose

This directory is the canonical runbook for generating the Kimi-authored
synthetic corpus.

The corpus has two separate objectives:

- `pretrain`: standalone documents for full next-token causal LM training.
- `sft`: XML-action chat rows for assistant-masked supervised training.

The long-run target is `500000000` generated tokens. The committed corpus lives
under `training/corpus/kimi-synthetic-v1/`.

## Read Order

1. [schema.md](schema.md): row formats, paths, and metadata.
2. [quality.md](quality.md): validation and scoring gates.
3. [workflow.md](workflow.md): sample-first and prompt-refinement flow.
4. [commands.md](commands.md): exact local commands.
5. [long-run.md](long-run.md): 500M-token background operation.

## Rules

- Use Kimi CLI non-interactively.
- Keep raw Kimi logs under `runs/kimi_corpus/logs/`.
- Do not paste full generated documents or ordinary CLI UI text into agent
  conversation.
- Validate before committing generated shards.
- Commit generated validated shards under `training/corpus/kimi-synthetic-v1/`.
- Keep `pretrain` and `sft` separated all the way through training.
