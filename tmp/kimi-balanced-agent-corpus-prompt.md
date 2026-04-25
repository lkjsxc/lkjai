You are Kimi Code working inside the `lkjai` repository.

Read these files before changing data:

- `docs/README.md`
- `docs/product/chat.md`
- `docs/product/api.md`
- `docs/architecture/agent/schema.md`
- `docs/architecture/agent/loop.md`
- `docs/architecture/training/corpus.md`
- `docs/architecture/training/provenance.md`
- `docs/architecture/training/pipeline.md`
- `docs/operations/training/agent-assessment.md`
- `docs/operations/training/kimi-corpus-generation.md`
- `training/lkjai_train/kimi_dataset.py`
- `training/lkjai_train/kimi_corpus*.py`
- `training/lkjai_train/rows.py`

Generate a high-quality balanced active corpus for `lkjai`.

Hard requirements:

- Output committed chunked JSONL under `training/corpus/kimi-full-v1/`.
- Use `train/`, `val/`, and `holdout/` split directories.
- Use about 1000 rows per chunk.
- Write `manifest.json` and `validation-report.json`.
- Target `500000000` train tokenizer tokens when feasible.
- Every assistant message must be exactly one XML action.
- Every active row must use `provenance=kimi-generated`,
  `author_type=external-agent-generated`, and `author_model=kimi-code`.
- Do not use GPT, Codex, Claude, or generic LLM-authored corpus packs as active
  training data.

Quality target:

- Keep the agent balanced: everyday conversation, XML validity, direct answers,
  tool use, observation handling, recovery, docs/source grounding, memory,
  preference handling, and kjxlkj confirmation flows.
- Ordinary chat such as `Hello`, `Thanks`, and `What can you help me with?`
  must finish directly with `agent.finish`.
- Ordinary chat must never call `fs.read`, `fs.list`, `shell.exec`, or kjxlkj
  tools.
- Ordinary chat must never answer with generic repository task-completion text
  such as `Completed task for docs/architecture/training/provenance.md.`
- Tool rows should teach when a tool is needed, how to revise after a failed
  observation, and when to stop rather than repeat an identical failed action.
- Use `<reasoning>` only for one short visible rationale. Do not include hidden
  chain-of-thought.

Required validation:

- Run corpus validation.
- Report row counts, split counts, token counts, duplicate rate, XML validity,
  final `agent.finish` rate, everyday-chat coverage, everyday-chat generic-final
  count, tool distribution, provenance distribution, source/license mix,
  commands run, blockers, and residual risks.
- Regenerate any shard that fails validation.
