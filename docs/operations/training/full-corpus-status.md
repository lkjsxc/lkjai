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

## Active Public-Pretrain Gap

The historical Kimi corpus is valid by its old checks but is no longer the
active 500M-token pretraining path. The active target is now
`data/public-corpus/` with permissive English public text and at least
`450000000` deduplicated train tokenizer tokens.
