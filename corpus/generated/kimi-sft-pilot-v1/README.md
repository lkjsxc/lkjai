# Kimi SFT Pilot V1

Generated through the authenticated Kimi CLI/API path on 2026-04-30.

## Contents

- `train/`: validated generated SFT shards.
- `val/`: validated generated validation shard.
- `holdout/`: validated generated holdout shard.
- `manifest.jsonl`: clean manifest containing only promoted valid shards.
- `validation-report.json`: deterministic score report for promoted shards.

Failed and quarantined staging shards were not promoted.

## Current Report

- Rows: `208`
- Approximate tokens: `25958`
- Supervised approximate tokens: `21325`
- Split rows: train `184`, val `12`, holdout `12`
- Template families: `52` rows each for direct finish, read-only retrieval,
  mutation confirmation, and failure/safety/recovery
- Validation flags: none
- Duplicate rate: `0.004807692307692308`
- Near-duplicate rate: `0.004807692307692308`

This is a real Kimi-generated pilot slice, not the full 1M-token pilot target.
The direct Moonshot HTTP keys available locally failed authentication; generation
used the authenticated Kimi CLI/API path.
