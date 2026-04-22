# Verification

## Goal

Verification proves:

- docs topology and line-limit constraints remain valid,
- runtime and training code paths still compile and pass tests,
- deterministic smoke checks still run inside the verification container.

## Mandatory Commands

```bash
docker compose --profile verify build verify
docker compose --profile verify run --rm verify
```

## Mandatory Checks in `verify.sh`

1. `cargo fmt -- --check`
2. `cargo test`
3. `python3 -m pytest training/tests`
4. `cargo run --bin lkjai -- docs validate-topology`
5. `cargo run --bin lkjai -- docs validate-links`
6. `cargo run --bin lkjai -- quality check-lines`
7. `cargo run --bin lkjai -- quality no-node`
8. `python3 -m lkjai_train.cli --data-dir "$DATA_DIR" smoke`

## Scope Boundary

- Verify is deterministic and lightweight compared with long GPU training runs.
- Verify does not prove final model quality by itself.
- Real training acceptance is governed by the training runbook and eval artifacts.
