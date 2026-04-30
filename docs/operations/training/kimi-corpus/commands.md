# Commands

## Setup

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py --help
python3 tools/kimi-corpus/score_corpus.py --help
```

## Fake Smoke

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --api-provider kimi-api \
  --target-tokens 50000 \
  --mode sft \
  --parallelism 2 \
  --output-dir /tmp/lkjai-kimi-fake \
  --dry-run \
  --max-calls 2
```

## Real Sample

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --api-provider kimi-api \
  --api-key-file /home/lkjsxc/private/password.md \
  --target-tokens 50000 \
  --mode sft \
  --parallelism 4 \
  --sample-first
```

## Score

```bash
python3 tools/kimi-corpus/score_corpus.py data/kimi_synthetic/samples \
  --markdown runs/kimi_corpus/sample_score.md \
  --output runs/kimi_corpus/sample_score.json
```

## Verify

```bash
docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

## Progress

```bash
python3 tools/kimi-corpus/score_corpus.py \
  corpus/generated/kimi-full-v1 \
  --summary-only
```

## Pilot

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

Then score the pilot:

```bash
python3 tools/kimi-corpus/score_corpus.py \
  corpus/generated/kimi-sft-pilot-v1 \
  --manifest corpus/generated/kimi-sft-pilot-v1/manifest.jsonl \
  --markdown runs/kimi_corpus/pilot_score.md \
  --output corpus/generated/kimi-sft-pilot-v1/validation-report.json
```

## Stop

```bash
bash tools/kimi-corpus/stop_background.sh --run-dir runs/kimi_corpus
```
