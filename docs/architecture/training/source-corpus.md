# Source Corpus

## Goal

Keep authored training content in LLM-readable JSON files instead of hiding it
inside Python constants.

## Location

- Source directory: `corpus/sources/`
- Source files: `*.json`
- Active public pretraining corpus: `data/public-corpus/`
- Active first-party SFT target: `60000000` XML-action tokens.
- Native loader: `native` trainer corpus reader.
- Generated dataset: `data/train/datasets/corpus.jsonl`

## File Shape

Each source file is one JSON array. Each element has tags and content:

```json
[
  {
    "tags": ["general_topic", "language:en"],
    "content": {
      "topic": "debugging",
      "practice": "reproduce, isolate, fix, verify",
      "failure": "premature guesses"
    }
  }
]
```

## Rules

- `tags` is a non-empty array of strings.
- `content` is a JSON object.
- Entries are human-reviewable and LLM-editable.
- Generation code may combine entries, but must not own large authored lists.
- JSON source files are commercial-safe project content unless a license entry
  states otherwise.
- Public-source entries must include license and source metadata.

## Active Source Files

- `public.json`: opt-in permissive public SFT allowlist.
- `public-pretrain.json`: active and reference English pretraining sources.
- SFT rows are generated from repo-derived content and committed JSONL shards.
- Product training reads generated rows through the native trainer.

## Removed Source Packs

Old LLM-authored JSON packs were deleted from the tracked tree. Active corpus
generation must use deterministic repo-derived modules, reviewed public-source
metadata, or validated generated corpus shards with explicit provenance.

## Public-Import Note

Public pretraining sources tagged `public_pretrain_dataset` are active recipes.
Raw data is downloaded or staged locally under `TRAIN_PUBLIC_DATA_DIR`;
generated large shards stay ignored under `data/public-corpus/`.
`prepare-public-pretrain` writes a validation report and fails unless staged
rows meet the requested token target.

Cosmopedia download page:
`https://huggingface.co/datasets/HuggingFaceTB/cosmopedia`

Default local layout:

```
data/raw/cosmopedia/          # user-downloaded Hugging Face snapshot
data/public-corpus/           # generated ignored train/val/holdout shards
```

Materialize the active 500M-token public pretraining corpus through native
training tooling:

```bash
docker compose --profile train up --build train
```

To activate a public source:

- Use only a mainline-allowed license: `Apache-2.0`, `MIT`, `BSD-2-Clause`, or
  `BSD-3-Clause`.
- Pin an immutable revision. Placeholder values such as `latest`, `main`,
  `master`, `pinned-by-operator`, and `review-required` are invalid for active
  sources.
- Normalize SFT sources into local `messages` JSONL before training.
- Normalize pretraining sources into local JSONL or Parquet with a `text` field.
- Use only Cosmopedia generated `text`; exclude `prompt` and `seed_data`.
- Preserve source URL, license, revision, row limit, skill, tags, and toolset.

Candidate sources under review include OASST1 English, OASST2 English,
smol-smoltalk, and Hermes Function-Calling V1. Dolly and xLAM remain
legal-review only under the conservative mainline policy.

Reference-only pretraining sources include FineWeb and Dolma because their
licenses are outside the current Apache/MIT/BSD policy.

## Rationale

- JSON arrays are easier for LLM agents to inspect, diff, and extend than Python
  constant blocks.
- JSONL remains the generated training artifact because it streams cleanly into
  native packed-cache construction.
- Tags make future filtering, provenance checks, and corpus balancing explicit.
