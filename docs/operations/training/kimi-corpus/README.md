# Kimi Corpus

## Purpose

This directory is the canonical runbook for generating the English-only
Kimi-authored synthetic corpus through the official Kimi API.

The active objective is `sft`: XML-action chat rows for assistant-masked
supervised training. Public Cosmopedia text supplies the causal-LM pretraining
side.

The first refreshed target is a smoke plus bounded pilot under
`corpus/generated/kimi-sft-pilot-v1/`. The full long-run target remains
`60000000` generated SFT tokens under `corpus/generated/kimi-sft-60m-v1/`, but
only after the pilot passes validation.

## Read Order

1. [schema.md](schema.md): row formats, paths, and metadata.
2. [quality.md](quality.md): validation and scoring gates.
3. [workflow.md](workflow.md): sample-first and prompt-refinement flow.
4. [commands.md](commands.md): exact local commands.
5. [long-run.md](long-run.md): 60M-token background operation.

## Rules

- Use the Kimi HTTP API, not the Kimi CLI, for active generation.
- Keep raw request metadata and redacted response logs under
  `runs/kimi_corpus/logs/`.
- Load API keys from `MOONSHOT_API_KEY`, `MOONSHOT_API_KEYS`, or an explicit
  local key file such as `/home/lkjsxc/private/password.md`.
- Treat every discovered key as usable in parallel and redact keys from all
  logs, manifests, and reports.
- Generate English rows only.
- Use public corpus projects as quality references, not copied text.
- Do not paste full generated documents or API logs into agent conversation.
- Validate before committing generated shards.
- Commit generated validated pilot shards under
  `corpus/generated/kimi-sft-pilot-v1/`.
- Commit full shards under `corpus/generated/kimi-sft-60m-v1/` only after pilot
  gates pass.
- Keep generator staging outputs separated by objective. The committed active
  full corpus is normalized into `train`, `val`, and `holdout` splits.
