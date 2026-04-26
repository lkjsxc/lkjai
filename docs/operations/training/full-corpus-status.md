# Full Corpus Status

## Generated Chunks

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

## Remaining Gap

The generated corpus more than doubles the previous 60k-row active corpus and
is chunked correctly, but it does not yet meet the 500M train-token target.
The next generator iteration must either stream many more rows or produce
longer high-quality scenarios without loading the entire corpus in memory.
