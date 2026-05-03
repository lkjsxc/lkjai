# Verification

## Goal

Verification proves:

- docs topology and line-limit constraints remain valid,
- runtime and training code paths still compile and pass tests,
- native CUDA checks run against the real GPU build.

## Mandatory Command

```bash
docker compose --progress quiet --profile verify run --rm verify
```

The existing `up --build --abort-on-container-exit verify` form remains useful
when inspecting Compose lifecycle behavior, but `run --rm verify` is the
canonical pass/fail gate.

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

The native CTest set includes config-contract, CUDA architecture policy,
capability-field, server unsupported-decode, packed-cache, report-schema, and
artifact/logits gates. Any missing additive capability field, missing Blackwell
`120` default arch flag, invalid profile config, accepted transformer report, or
chat response with `choices` fails the gate.

## Compact Logs

`ops/verify.sh` writes full command logs under `/tmp/lkjai-verify-logs` inside the
container and prints only one pass line per check. On failure it prints the last
`VERIFY_TAIL_LINES`, default `120`, from the failing log.

Use this when an agent needs the failure without reading full Docker logs:

```bash
VERIFY_TAIL_LINES=80 docker compose --progress quiet --profile verify run --rm verify
```

## Scope Boundary

- Verify requires NVIDIA GPU access, but remains bounded compared with long
  training runs.
- Verify does not prove final model quality by itself.
- Scratch training acceptance is governed by the training runbook and eval
  artifacts.
