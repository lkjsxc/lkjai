# Training Corpus

## Goal

Define the canonical license-conservative training pack for scratch language
modeling and XML-action supervision.

## Storage Schema

Editable source entries live in JSON array files under `corpus/sources/`; see
[source-corpus.md](source-corpus.md). SFT rows use `lkjai-agent-jsonl-v2`:

```json
{
  "messages": [
    {"role": "user", "content": "<task><request>Summarize ...</request></task>"},
    {"role": "assistant", "content": "<action>\n<tool>agent.finish</tool>\n<content>...</content>\n</action>"}
  ],
  "tags": ["docs_grounding", "language:en"],
  "meta": {
    "id": "docs-lkjai-000001",
    "split": "train",
    "provenance": "repo-derived",
    "author_type": "repo-derived",
    "author_model": "none",
    "quality_tier": "high",
    "domain": "lkjai-docs",
    "skill": "grounding",
    "toolset": "none",
    "language": "en",
    "safety_scope": "workspace-safe",
    "license": "project-local",
    "source_ref": "docs/architecture/training/corpus.md"
  }
}
```

Public pretraining rows use standalone English `text` with metadata:

```json
{
  "id": "public-pretrain-cosmopedia-stanford-000000001",
  "mode": "pretrain",
  "language": "en",
  "domain": "cosmopedia-stanford",
  "text": "Generated educational text...",
  "metadata": {
    "provenance": "public-pretrain",
    "license": "Apache-2.0",
    "field_policy": "text-only",
    "excluded_fields": ["prompt", "seed_data"]
  }
}
```

## Rules

- `messages`, `tags`, and `meta` are required for SFT rows.
- Roles allowed: `system`, `user`, `assistant`, `tool`.
- `assistant` content must be one valid XML action.
- Public rows must use explicit permissive licenses only.
- The mainline corpus must stay commercial-safe.
- Default rows must not contain GPT/Codex-authored corpus text.
- Public pretraining rows must declare pinned source, license, selected field
  policy, and `public-pretrain` provenance.
- Public pretraining rows must not include `prompt` or `seed_data` values.
- Quarantined source packs must not be consumed by `prepare-corpus`.

## Model-Facing Serialization

- Storage remains JSONL.
- Model-facing text uses paired XML-like sections.
- Prompt construction ends with `<assistant_action>` so the model learns the same
  continuation boundary used during inference.
- The boundary includes the newline after `<assistant_action>`; inference must
  preserve that exact next-token context.
- Assistant outputs use one XML action whose child tags become typed tool fields.

## Dataset Targets

- Complete release pack target: `500000000` tokens.
- Public pretraining target: `440000000` generated `text`-only tokens.
- First-party XML-action SFT target: `60000000` tokens.
- Duplicate rows: at most `1%`.
- Deduplicated tokenizer tokens in the complete release pack: at least
  `450000000`.
- Active public corpora must use `Apache-2.0`, `MIT`, `BSD-2-Clause`, or
  `BSD-3-Clause`.

## Token Budget

| Component | Target tokens | Objective |
|---|---:|---|
| Cosmopedia `stanford` `text` only | `180000000` | `causal_lm_full` |
| Cosmopedia `wikihow` `text` only | `90000000` | `causal_lm_full` |
| Cosmopedia `openstax` `text` only | `50000000` | `causal_lm_full` |
| Cosmopedia `khanacademy` `text` only | `20000000` | `causal_lm_full` |
| Cosmopedia `auto_math_text` `text` only | `90000000` | `causal_lm_full` |
| Cosmopedia `stories` `text` only | `10000000` | `causal_lm_full` |
| First-party XML-action traces | `60000000` | `assistant_masked_sft` |

The public side teaches English fluency and explanation style. The first-party
side teaches action XML, tool choice, memory, recovery, and `kjxlkj` behavior.

## Public Pretraining Layout

Ignored generated pretraining chunks live under:

```
data/public-corpus/
  manifest.json
  validation-report.json
  train/train-000001.jsonl
  val/val-000001.jsonl
  holdout/holdout-000001.jsonl
```

Raw public downloads stay outside git under:

```
data/raw/cosmopedia/
```

The active public source is the Apache-2.0 Cosmopedia Hugging Face dataset:
`https://huggingface.co/datasets/HuggingFaceTB/cosmopedia`

Materialize ignored shards with:

```bash
HF_TOKEN=<hugging-face-token-if-needed> \
docker compose --profile corpus run --rm corpus download-public-pretrain

TRAIN_PUBLIC_DATA_DIR=/app/data/raw/cosmopedia \
TRAIN_CORPUS_DIR=/app/data/public-corpus \
docker compose --profile corpus run --rm corpus prepare-public-pretrain
```

`manifest.json` records schema, row counts, split counts, token counts, selected
fields, source budgets, checksums, and source/license distribution.

## Rejection Patterns

- Everyday-chat rows must not finish with `Completed task for ...`.
- Greetings, thanks, and capability questions must not call filesystem tools.
- Repeated failed tool calls must be represented as failures to avoid, not as
  successful target behavior.
- Generic final answers belong only in separate preference-pair artifacts.
- Rejected alternatives and preference-pair rows must not appear as ordinary
  assistant targets in the active corpus.
