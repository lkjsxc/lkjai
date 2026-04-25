# Kimi Corpus Generation

## Goal

Generate and commit the full Kimi-authored XML action corpus for the scratch
agent, with enough everyday conversation coverage to make basic chat usable.

## Target

- Train tokenizer tokens: `500000000`.
- Train tokenizer tokens after dedupe: at least `450000000`.
- Chunk size: about `1000` JSONL rows.
- Committed location: `training/corpus/kimi-full-v1/`.
- Runtime staging location: `data/kimi-corpus/`.

## Mix

- Everyday conversation: `50%`.
- Reasoning, planning, and clarification: `20%`.
- Local tool use and recovery: `15%`.
- Docs and source grounding: `10%`.
- kjxlkj read and confirmation flows: `5%`.

## Quality Gates

- XML validity rate: `>= 0.995`.
- Last assistant action is `agent.finish`: `1.0`.
- Duplicate rate: `<= 0.01`.
- Generic final-answer rate: `<= 0.005`.
- Each split has everyday conversation rows.
- Non-final chunks have roughly `1000` rows.
- Every active row has `kimi-generated` provenance.

## KimiCode Prompt

Copy the prompt below into Kimi Code.

```text
You are Kimi Code working inside the lkjai repository.

Read docs/README.md, docs/architecture/agent/schema.md,
docs/architecture/agent/loop.md, docs/architecture/training/corpus.md,
docs/architecture/training/provenance.md, docs/architecture/training/pipeline.md,
docs/operations/training/agent-assessment.md, and the training/lkjai_train
corpus and dataset modules before changing anything.

Generate the active full training corpus for lkjai:
- target 500000000 train tokenizer tokens,
- commit chunked JSONL under training/corpus/kimi-full-v1,
- use about 1000 rows per chunk,
- use train, val, and holdout split directories,
- write manifest.json and validation-report.json,
- prioritize everyday conversation in 50% of rows,
- include reasoning/planning, local tools, docs/source grounding, and kjxlkj
  confirmation flows in the remaining rows.

Every assistant message must be exactly one XML action:
<action>
<reasoning>brief visible rationale</reasoning>
<tool>agent.finish</tool>
<content>Final answer.</content>
</action>

Use <reasoning> as a short visible rationale only. Do not write long hidden
chain-of-thought. Use agent.finish directly for simple greetings, thanks,
clarifications, concise answers, and normal everyday chat. Use agent.think only
for explicit non-terminating planning. Never emit JSON assistant actions.

Rows must use:
- provenance: kimi-generated
- author_type: external-agent-generated
- author_model: kimi-code
- explicit source_ref
- explicit license

Do not use GPT, Codex, Claude, or generic LLM-authored corpus packs as active
training data. Regenerate any shard that fails validation.

Run validation and report row counts, split counts, token counts, duplicate
rate, XML validity, agent.finish rate, chunk sizes, everyday-chat coverage,
tool distribution, provenance distribution, source/license mix, commands run,
blockers, and residual risks.
```
