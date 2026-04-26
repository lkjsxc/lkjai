# Long Run

## Launch

Run the full generation after sample quality and verification pass:

```bash
bash tools/kimi-corpus/launch_background.sh \
  --config configs/corpus/kimi_500m.yaml \
  --target-tokens 500000000 \
  --parallelism 2 \
  --output-dir corpus/generated/kimi-full-v1 \
  --full
```

## Controls

- `--parallelism 2` runs roughly two Kimi calls at a time.
- `--target-tokens 500000000` is the default full target.
- `--resume` skips valid completed shards.
- `runs/kimi_corpus/STOP` requests graceful stop.
- `runs/kimi_corpus/kimi_corpus.pid` records the launcher PID.

## Commit Cadence

Commit generated shards after validation:

- every 5M valid approximate tokens, or
- every 100 valid shards,
- and whenever stopping due to quota, timeout, or operator intervention.

## Risks

- The run can take a long time.
- Subscription or rate limits may pause generation.
- Synthetic-only corpora can become repetitive.
- A large committed corpus increases repository size.

If generation stops early, keep valid shards, update
`corpus/generated/kimi-full-v1/README.md`, and commit the current status.
