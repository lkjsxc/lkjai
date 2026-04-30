# Verification

## Goal

Verification proves:

- docs topology and line-limit constraints remain valid,
- runtime and training code paths still compile and pass tests,
- default checks stay lightweight enough for frequent Compose runs.

## Mandatory Command

```bash
docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

## Mandatory Checks in `ops/verify.sh`

1. `cargo fmt -- --check`
2. `cargo test`
3. `cmake -S native -B /tmp/lkjai-native-build -G Ninja`
4. `cmake --build /tmp/lkjai-native-build --parallel`
5. `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`
6. `cargo run --bin lkjai -- docs validate-topology`
7. `cargo run --bin lkjai -- docs validate-links`
8. `cargo run --bin lkjai -- quality check-lines`
9. `cargo run --bin lkjai -- quality no-node`

## Compact Logs

`ops/verify.sh` writes full command logs under `/tmp/lkjai-verify-logs` inside the
container and prints only one pass line per check. On failure it prints the last
`VERIFY_TAIL_LINES`, default `120`, from the failing log.

Use this when an agent needs the failure without reading full Docker logs:

```bash
VERIFY_TAIL_LINES=80 docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

## Scope Boundary

- Verify is deterministic and lightweight compared with long GPU training runs.
- Verify does not prove final model quality by itself.
- Scratch training acceptance is governed by the training runbook and eval
  artifacts.
