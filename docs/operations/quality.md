# Quality Gates

## Mandatory Gates

1. `cmake -S native -B /tmp/lkjai-native-build -G Ninja`
2. `cmake --build /tmp/lkjai-native-build --parallel`
3. `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`
4. `lkjai-native-repo-check docs-topology --repo /workspace`
5. `lkjai-native-repo-check docs-links --repo /workspace`
6. `lkjai-native-repo-check docs-contract-owners --repo /workspace`
7. `lkjai-native-repo-check corpus-actions -- FILE...`
8. `lkjai-native-repo-check config-contract --repo /workspace`
9. `lkjai-native-repo-check cuda-arch-contract --repo /workspace`
10. `lkjai-native-repo-check line-limits --repo /workspace`
11. `lkjai-native-repo-check no-node --repo /workspace`
12. `lkjai-native-repo-check native-only --repo /workspace`

## Compose Gate

```bash
docker compose --progress quiet --profile verify run --build --rm verify
```

`ops/verify.sh` keeps full logs in `/tmp/lkjai-verify-logs` and tails only failing
checks by default. Set `VERIFY_TAIL_LINES` to tune failure output size.

## Training Gate

- `docker compose --profile train up --build train` runs the committed smoke
  command and is not a full long-training gate.
- Full training gates require an explicit `--train` command, matching config,
  packed cache, report, and eval artifacts.
- The bounded Docker start check in
  [training/long-run.md](training/long-run.md) must pass for training-config
  changes.
- It produces `runs/fixed-eval.json` and `runs/behavioral-eval.json`.
- Fixed eval acceptance requires XML-action artifacts to pass configured gates.
- Current accepted behavioral baseline is none; latest raw repair runs remain
  `pass_rate=0.0`.
- The next improvement gate is `TRAIN_BEHAVIORAL_THRESHOLD`, default `0.35`.
- Public-pretrain validation reports must be updated for runs that change the
  500M public pretraining corpus.
- Agent competency acceptance remains behavioral `pass_rate >= 0.80`.
- For strict enforcement, keep `TRAIN_ENFORCE_COMPETENCY=1`.
- For exploratory runs, override with `TRAIN_ENFORCE_COMPETENCY=0`.
- Real model quality must be judged from generated XML actions, real tool
  execution, `agent.finish` responses, and behavioral reports.

## Stop Rule

- Any non-zero mandatory gate blocks acceptance.
