# Quality Gates

## Mandatory Gates

1. `cargo fmt -- --check`
2. `cargo test`
3. `python3 -m pytest -m "not slow" training/tests`
4. `cargo run --bin lkjai -- docs validate-topology`
5. `cargo run --bin lkjai -- docs validate-links`
6. `cargo run --bin lkjai -- quality check-lines`
7. `cargo run --bin lkjai -- quality no-node`

## Compose Gate

```bash
docker compose --profile verify up --build --abort-on-container-exit verify
```

## Training Gate

- `docker compose --profile train up --build train` is a separate training gate.
- It produces `runs/fixed-eval.json` and `runs/behavioral-eval.json`.
- Fixed eval acceptance requires every artifact case to pass.
- Current behavioral baseline is raw holdout `pass_rate=0.235`.
- The next improvement gate is `TRAIN_BEHAVIORAL_THRESHOLD`, default `0.35`.
- Agent competency acceptance remains behavioral `pass_rate >= 0.80`.
- For strict enforcement, keep `TRAIN_ENFORCE_COMPETENCY=1`.
- For exploratory runs, override with `TRAIN_ENFORCE_COMPETENCY=0`.
- Real model quality must be judged from generated responses and behavioral
  reports, not docs-only verification.

## Stop Rule

- Any non-zero mandatory gate blocks acceptance.
