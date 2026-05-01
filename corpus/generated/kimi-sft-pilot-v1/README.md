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

- Rows: `784`
- Approximate tokens: `100523`
- Supervised approximate tokens: `82400`
- Split rows: train `688`, val `48`, holdout `48`
- Template families: `196` rows each for direct finish, read-only retrieval,
  mutation confirmation, and failure/safety/recovery
- Gold stop reasons: `588` finish rows, `196` confirmation-required rows
- Resource tool rows: `31` search, `2` fetch, `2` history, `1` preview
- Validation flags: none
- Duplicate rate: `0.006377551020408163`
- Near-duplicate rate: `0.008928571428571428`

This is a real Kimi-generated pilot slice, not the full 1M-token pilot target.
The direct Moonshot HTTP keys available locally failed authentication; generation
used the authenticated Kimi CLI/API path.
