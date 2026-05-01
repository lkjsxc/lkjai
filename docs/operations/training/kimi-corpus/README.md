# Kimi Corpus

## Purpose

This directory is the canonical runbook for generating the English-only
Kimi-authored synthetic corpus through the official Kimi API.

The active objective is `sft`: XML-action chat rows for assistant-masked
supervised training. Public Cosmopedia text supplies the causal-LM pretraining
side.

The accepted refreshed target is `60000000` generated SFT tokens under
`corpus/generated/kimi-sft-60m-v2/`. A bounded pilot may be staged under
ignored `data/kimi_synthetic/`, but no pilot is active unless it passes the
same runtime validator and fixture gates as the full corpus.

## Read Order

1. [schema.md](schema.md): row formats, paths, and metadata.
2. [quality.md](quality.md): validation and scoring gates.
3. [workflow.md](workflow.md): sample-first and prompt-refinement flow.
4. [commands.md](commands.md): exact local commands.
5. [long-run.md](long-run.md): 60M-token background operation.

## Rules

- Use the Kimi HTTP API, not the Kimi CLI, for active generation.
- Request JSON object output with a top-level `rows` array; local validation is
  the contract owner.
- Keep raw request metadata and redacted response logs under
  `runs/kimi_corpus/logs/`.
- Load API keys from `MOONSHOT_API_KEY`, `MOONSHOT_API_KEYS`, or an explicit
  local key file such as
  `/home/lkjsxc/private/archived/security/password.md`.
- Treat every discovered key as usable in parallel and redact keys from all
  logs, manifests, and reports.
- Generate English rows only.
- Use public corpus projects as quality references, not copied text.
- Do not paste full generated documents or API logs into agent conversation.
- Validate before committing generated shards.
- Commit generated validated SFT shards under
  `corpus/generated/kimi-sft-60m-v2/`.
- Commit preference pairs under `corpus/generated/pref-v1/`.
- Keep generator staging outputs separated by objective. The committed active
  full corpus is normalized into `train`, `val`, and `holdout` splits.
