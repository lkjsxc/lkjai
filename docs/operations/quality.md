# Quality Gates

## Mandatory Gates

1. `cargo fmt -- --check`
2. `cargo test`
3. `cmake -S native -B /tmp/lkjai-native-build -G Ninja`
4. `cmake --build /tmp/lkjai-native-build --parallel`
5. `ctest --test-dir /tmp/lkjai-native-build --output-on-failure`
6. `cargo run --bin lkjai -- docs validate-topology`
7. `cargo run --bin lkjai -- docs validate-links`
8. `cargo run --bin lkjai -- quality check-lines`
9. `cargo run --bin lkjai -- quality no-node`

## Compose Gate

```bash
docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

`ops/verify.sh` keeps full logs in `/tmp/lkjai-verify-logs` and tails only failing
checks by default. Set `VERIFY_TAIL_LINES` to tune failure output size.

## Training Gate

- `docker compose --profile train up --build train` is the full long training
  gate and is not required for ordinary code verification.
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
