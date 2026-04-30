# Workflow

## Order

1. Read this directory and the training architecture docs.
2. Run a dry-run smoke test.
3. Resolve models, balance, and API token estimates.
4. Generate a real sample with structured Kimi API output.
5. Score the sample and write `runs/kimi_corpus/sample_report.md`.
6. Refine scenarios or prompts using only aggregate scores and short excerpts.
7. Regenerate a second sample.
8. Score the second sample.
9. Generate the 1M-token pilot only after sample quality is acceptable.
10. Launch the full long run only after pilot quality is acceptable.

## Prompt Refinement

Prompt refinement must preserve:

- strict JSONL-only output,
- required fields,
- original synthetic content requirement,
- no copyrighted reproduction,
- no benchmark dumps,
- no assistant-chat framing in pretraining,
- assistant-masked SFT row compatibility.
- the four active template families,
- confirmation-before-mutation labels.

Do not blindly accept a refined prompt. Validate the prompt text before saving a
new `sft_vN.txt`.

## Logs

Redacted Kimi request metadata and responses stay in `runs/kimi_corpus/logs/`.

Reports may include:

- aggregate score summaries,
- flag counts,
- short representative excerpts,
- command paths,
- model names,
- redacted key fingerprints,
- blocker summaries.

Reports must not include API keys, full Kimi logs, or full generated documents.

## Commits

Commit coherent batches:

1. docs updates,
2. generator and scoring code,
3. tests and fake fixtures,
4. validated sample reports,
5. validated generated shard batches.
