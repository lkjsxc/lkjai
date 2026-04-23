# Verification

## Goal

Verification proves:

- docs topology and line-limit constraints remain valid,
- runtime and training code paths still compile and pass tests,
- default checks stay lightweight enough for frequent Compose runs.

## Mandatory Command

```bash
docker compose --profile verify up --build --abort-on-container-exit verify
```

## Mandatory Checks in `verify.sh`

1. `cargo fmt -- --check`
2. `cargo test`
3. `python3 -m pytest -o cache_dir=/tmp/pytest-cache -m "not slow" training/tests`
4. `cargo run --bin lkjai -- docs validate-topology`
5. `cargo run --bin lkjai -- docs validate-links`
6. `cargo run --bin lkjai -- quality check-lines`
7. `cargo run --bin lkjai -- quality no-node`

## Scope Boundary

- Verify is deterministic and lightweight compared with long GPU training runs.
- Verify does not prove final model quality by itself.
- Scratch training acceptance is governed by the training runbook and eval
  artifacts.
