# Verification

## Goal

Verification proves:

- docs topology and line-limit constraints remain valid,
- runtime and training code paths still compile and pass tests,
- native CUDA checks run against the real GPU build.

## Mandatory Command

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```

The existing `up --build --abort-on-container-exit verify` form remains useful
when inspecting Compose lifecycle behavior, but `run --rm verify` is the
canonical pass/fail gate.

## Mandatory Checks in `ops/verify.sh`

1. `cmake -S native -B /tmp/lkjai-native-build -G Ninja`
2. `cmake --build /tmp/lkjai-native-build --parallel`
3. `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`
4. `lkjai-native-repo-check docs-topology --repo /workspace`
5. `lkjai-native-repo-check docs-links --repo /workspace`
6. `lkjai-native-repo-check corpus-actions -- FILE...`
7. `lkjai-native-repo-check config-contract --repo /workspace`
8. `lkjai-native-repo-check cuda-arch-contract --repo /workspace`
9. `lkjai-native-repo-check line-limits --repo /workspace`
10. `lkjai-native-repo-check no-node --repo /workspace`

The native CTest set includes config-contract, CUDA architecture policy,
capability-field, dense/transformer unsupported-decode, decoder chat,
packed-cache, report-schema, and artifact/logits gates. Any missing additive
capability field, missing Blackwell `120` default arch flag, invalid profile
config, or accepted transformer report fails the gate.

## Compact Logs

`ops/verify.sh` writes full command logs under `/tmp/lkjai-verify-logs` inside the
container and prints only one pass line per check. On failure it prints the last
`VERIFY_TAIL_LINES`, default `120`, from the failing log.

Use this when an agent needs the failure without reading full Docker logs:

```bash
VERIFY_TAIL_LINES=80 docker compose --progress quiet --profile verify run --build --rm verify
```

## Scope Boundary

- Verify requires NVIDIA GPU access, but remains bounded compared with long
  training runs.
- Verify does not prove final model quality by itself.
- Scratch training acceptance is governed by the training runbook and eval
  artifacts.
