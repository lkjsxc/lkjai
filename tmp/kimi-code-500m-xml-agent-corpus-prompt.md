# Kimi Code Prompt: Build The 500M-Token XML Agent Corpus

Copy everything below this line into Kimi Code.

---

You are Kimi Code working inside the `lkjai` repository. Create the active
training corpus and supporting generators for the new XML-action agent.

## Mission

Generate a commercial-safe, high-quality, multi-turn agent corpus targeting:

- `500000000` train tokenizer tokens.
- 60M-parameter scratch model training.
- XML assistant actions, not JSON actions.
- Real tool-use traces ending with `agent.finish`.
- Active Kimi provenance.

Do not use Codex/GPT-authored corpus text as active training data.

## Read First

Read these files before changing anything:

- `docs/README.md`
- `docs/architecture/agent/schema.md`
- `docs/architecture/agent/loop.md`
- `docs/architecture/training/corpus.md`
- `docs/architecture/training/provenance.md`
- `docs/architecture/training/pipeline.md`
- `docs/research/prompt-format.md`
- `training/lkjai_train/rows.py`
- `training/lkjai_train/dataset.py`
- `training/lkjai_train/formatting.py`
- `training/lkjai_train/settings.py`

## Required Action Format

Every assistant message must be exactly one XML action:

```xml
<action>
<reasoning>private concise reasoning</reasoning>
<tool>fs.read</tool>
<path>docs/README.md</path>
</action>
```

Successful turns must terminate through:

```xml
<action>
<reasoning>the task is complete</reasoning>
<tool>agent.finish</tool>
<content>Final user-facing answer.</content>
</action>
```

Use `<tool>agent.think</tool>` only for explicit non-terminating planning steps.
Do not emit JSON assistant actions.

## Provenance Metadata

Rows you generate for the active corpus must use:

- `provenance`: `kimi-generated`
- `author_type`: `external-agent-generated`
- `author_model`: `kimi-code`
- explicit `source_ref`
- explicit `license`

Do not mark Kimi rows as `repo-derived` unless the row is mechanically derived
from repository files without invented content.

## Storage Target

Do not commit the full corpus to git.

Write generated shards under:

```text
data/kimi-corpus/
```

Use JSONL as the container format because the project loader expects JSONL.
Inside each row, assistant content must be XML.

Required outputs:

- `data/kimi-corpus/train/*.jsonl`
- `data/kimi-corpus/val/*.jsonl`
- `data/kimi-corpus/holdout/*.jsonl`
- `data/kimi-corpus/manifest.json`
- `data/kimi-corpus/validation-report.json`
- generator code and docs updates committed to git

## Corpus Mix

Target a high-quality blend:

- 25% repository/docs grounded tool-use traces.
- 20% multi-turn implementation and debugging traces.
- 15% runtime tool schema and argument selection traces.
- 15% failed-tool recovery and environment blocker traces.
- 10% confirmation and safe mutation flows.
- 10% preference, critique, and revision traces.
- 5% licensed public-import only if licensing is pinned.

Prefer depth over shallow paraphrase multiplication.

## Quality Rules

- Every multi-turn task must include real-looking observations.
- Never let the assistant invent a tool result after requesting a tool.
- A tool result must appear as a tool message before the model uses it.
- Every successful trace ends with `agent.finish`.
- Train/val/holdout must be isolated by scenario template and source.
- Duplicate normalized rows must stay below 1%.
- Holdout must include hard cases, not just easy direct answers.
- Use private `<reasoning>` sparingly but enough to teach deeper thinking.

## Validation

Implement or update validators so they report:

- total rows
- train/val/holdout rows
- tokenizer token count
- duplicate rate
- XML validity
- action tool distribution
- termination rate through `agent.finish`
- provenance distribution
- source/license distribution

Required checks:

```bash
python3 -m compileall -q training/lkjai_train training/tests
PYTHONPATH=training python3 -m lkjai_train.cli validate-sources
PYTHONPATH=training python3 -m lkjai_train.cli --data-dir /tmp/lkjai-kimi-smoke prepare-corpus
PYTHONPATH=training python3 -m lkjai_train.cli --data-dir /tmp/lkjai-kimi-smoke validate-dataset
docker compose --profile verify up --build --abort-on-container-exit verify
```

If Docker or dependencies are unavailable, record the exact blocker and run all
checks that work.

## Commit Discipline

Commit coherent milestones:

1. Corpus generator and manifest support.
2. Validator and tests.
3. Documentation updates.
4. Smoke generation report.

## Final Report

Report:

- commit SHAs
- generated token count
- shard paths
- split counts
- duplicate rate
- XML validity rate
- `agent.finish` termination rate
- provenance/source/license mix
- validation commands and results
- blockers and residual risks

Begin by reading docs and current implementation, then build.
