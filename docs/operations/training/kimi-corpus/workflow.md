# Workflow

## Order

1. Read this directory and the training architecture docs.
2. Run fake Kimi smoke tests.
3. Generate a real sample with Kimi.
4. Score the sample and write `runs/kimi_corpus/sample_report.md`.
5. Ask Kimi to refine prompts using only aggregate scores and short excerpts.
6. Regenerate a second sample.
7. Score the second sample.
8. Launch the long run only after sample quality is acceptable.
9. Periodically score, commit validated shards, and update the corpus README.

## Prompt Refinement

Prompt refinement must preserve:

- strict JSONL-only output,
- required fields,
- original synthetic content requirement,
- no copyrighted reproduction,
- no benchmark dumps,
- no assistant-chat framing in pretraining,
- assistant-masked SFT row compatibility.

Do not blindly accept a refined prompt. Validate the prompt text before saving a
new `pretrain_vN.txt` or `sft_vN.txt`.

## Logs

Raw Kimi stdout and stderr stay in `runs/kimi_corpus/logs/`.

Reports may include:

- aggregate score summaries,
- flag counts,
- short representative excerpts,
- command paths,
- blocker summaries.

Reports must not include full Kimi logs or full generated documents.

## Commits

Commit coherent batches:

1. docs updates,
2. generator and scoring code,
3. tests and fake fixtures,
4. validated sample reports,
5. validated generated shard batches.
