# Long Run

## Launch

Run the pilot after sample quality and verification pass:

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_sft_60m.yaml \
  --api-provider kimi-api \
  --api-key-file /home/lkjsxc/private/password.md \
  --target-tokens 1000000 \
  --mode sft \
  --parallelism 8 \
  --output-dir corpus/generated/kimi-sft-pilot-v1 \
  --run-dir runs/kimi_corpus \
  --resume
```

## Controls

- Parallelism may use all discovered API keys when rate-limit errors stay below
  the retry budget.
- `--target-tokens 60000000` is the default SFT target.
- `--resume` skips valid completed shards.
- `runs/kimi_corpus/STOP` requests graceful stop.
- API keys are read from environment variables or the explicit local key file.
- Logs and manifests must contain only redacted key fingerprints.

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
`corpus/generated/kimi-sft-pilot-v1/README.md`, and commit the current status.
After pilot acceptance, repeat with `--target-tokens 60000000` and
`--output-dir corpus/generated/kimi-sft-60m-v1`.
