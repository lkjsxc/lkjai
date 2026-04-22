# Quality Gates

## Mandatory Gates

1. `cargo fmt -- --check`
2. `cargo test`
3. `python3 -m pytest training/tests`
4. `cargo run --bin lkjai -- docs validate-topology`
5. `cargo run --bin lkjai -- docs validate-links`
6. `cargo run --bin lkjai -- quality check-lines`
7. `python3 -m lkjai_train.cli smoke`
8. `cargo run --bin lkjai -- quality no-node`

## Compose Gate

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Long-Run Acceptance Gate

- `docker compose --profile train up --build train` produces `runs/fixed-eval.json`.
- Agent competency acceptance requires `pass_rate >= 0.80`.
- For strict enforcement, keep `TRAIN_ENFORCE_COMPETENCY=1`.
- For exploratory runs, override with `TRAIN_ENFORCE_COMPETENCY=0`.
- Real model quality must be judged from trained adapter artifacts + eval report,
  not from verify-only smoke checks.

## Stop Rule

- Any non-zero gate blocks acceptance.
