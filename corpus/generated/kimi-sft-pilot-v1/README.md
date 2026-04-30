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

- Rows: `68`
- Approximate tokens: `8583`
- Supervised approximate tokens: `7185`
- Split rows: train `60`, val `4`, holdout `4`
- Template families: `17` rows each for direct finish, read-only retrieval,
  mutation confirmation, and failure/safety/recovery
- Validation flags: none

This is a real Kimi-generated pilot slice, not the full 1M-token pilot target.
The direct Moonshot HTTP keys available locally failed authentication; generation
used the authenticated Kimi CLI/API path.
