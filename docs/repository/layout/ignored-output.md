# Ignored Output

Owner: `docs/repository/layout/ignored-output.md`.
State: canonical documentation.

## Local Runtime Output

- `data/models/`: served model artifacts.
- `data/train/`: tokenizer, packed cache, checkpoints, exports, and reports.
- `data/corpus/quarantine/`: generated corpus candidates awaiting validation.
- `data/corpus/generated/`: promoted generated corpora such as full Kimi SFT.
- `data/agent/`: runtime transcripts and agent state.
- `data/raw/` and `data/public-corpus/`: acquired public corpus material.
- `data/verify/`: verification fixtures and generated outputs.

## Temporary Evidence

- `tmp/`: ignored research reports, work notes, and transient evidence.
- Reports under `tmp/` are source inputs only.
- Durable conclusions from `tmp/` must be summarized into `docs/` under the
  line-limit and topology rules.

## Secret Paths

- `data/secrets/`: local secret files such as Hugging Face tokens.
- Docs and committed config may mention secret paths or environment variable
  names, but must never include token values.
