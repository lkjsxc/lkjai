# Quality Gates

## Mandatory Gates

1. `cargo fmt -- --check`
2. `cargo test`
3. `python -m pytest training/tests`
4. `cargo run --bin lkjai -- docs validate-topology`
5. `cargo run --bin lkjai -- docs validate-links`
6. `cargo run --bin lkjai -- quality check-lines`
7. `python -m lkjai_train.cli smoke`
8. `cargo run --bin lkjai -- quality no-node`

## Compose Gate

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Stop Rule

- Any non-zero gate blocks acceptance.
