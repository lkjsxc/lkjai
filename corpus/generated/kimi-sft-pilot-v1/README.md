# Kimi SFT Pilot V1

Generated through the authenticated Kimi CLI/API path on 2026-05-01.

## Contents

- `train/`: validated generated SFT shards.
- `val/`: validated generated validation shard.
- `holdout/`: validated generated holdout shard.
- `manifest.jsonl`: clean manifest containing only promoted valid shards.
- `validation-report.json`: deterministic score report for promoted shards.

Failed and quarantined staging shards were not promoted.

## Current Report

- Rows: `400`
- Approximate tokens: `50809`
- Supervised approximate tokens: `41729`
- Split rows: train `348`, val `28`, holdout `24`
- Template families: `100` rows each for direct finish, read-only retrieval,
  mutation confirmation, and failure/safety/recovery
- Gold stop reasons: `300` finish rows, `100` confirmation-required rows
- Resource tool rows: `13` search, `2` fetch, `2` history
- Validation flags: none
- Duplicate rate: `0.0025`
- Near-duplicate rate: `0.005`

This is a real Kimi-generated pilot slice, not the full 1M-token pilot target.
The direct Moonshot HTTP keys available locally failed authentication; generation
used the authenticated Kimi CLI/API path.
