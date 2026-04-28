# Full Corpus Status

## Historical Kimi Chunks

- Path: `corpus/generated/kimi-full-v1/`.
- Rows: `176345`.
- Train rows: `141258`.
- Validation rows: `17539`.
- Holdout rows: `17548`.
- JSONL chunks: `178`.
- Disk size: about `180M`.

## Validation Report

- Report: `corpus/generated/kimi-full-v1/validation-report.json`.
- Token count reported by local validator: `25962822`.
- Duplicate rate: `0.0`.
- XML validity rate: `1.0`.
- `agent.finish` termination rate: `1.0`.
- Everyday-chat rows: `125001`.
- Generic final-answer rate: about `0.000011`.

## Active Release-Pack Gap

The historical Kimi corpus is valid by its old checks but is no longer the
active full training pack. The active target is now:

- `500000000` Cosmopedia `text`-only public pretraining tokens under
  `data/public-corpus/`.
- `60000000` first-party XML-action SFT tokens.
- At least `450000000` deduplicated tokenizer tokens across the release pack.

## Public Pretrain Materialization

Current generated public corpus:

- Path: `data/public-corpus/`.
- Manifest: `data/public-corpus/manifest.json`.
- Target tokens: `500000000`.
- Approximate tokens: pending rebuild.
- Rows: pending rebuild.
- Duplicate rows: pending rebuild.
- Duplicate rate: pending rebuild.
- Source mix:
  - `cosmopedia-stanford`: `180000719`.
  - `cosmopedia-wikihow`: `90001277`.
  - `cosmopedia-auto-math-text`: `89997320`.
  - `cosmopedia-openstax`: `50000260`.
  - `cosmopedia-khanacademy`: `20000634`.
  - `cosmopedia-stories`: `70000000` target.
