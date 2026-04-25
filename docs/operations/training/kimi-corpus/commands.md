# Commands

## Setup

```bash
kimi --help
kimi info
python3 scripts/kimi_corpus/generate_kimi_corpus.py --help
python3 scripts/kimi_corpus/score_corpus.py --help
```

## Fake Smoke

```bash
python3 scripts/kimi_corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --target-tokens 50000 \
  --mode mixed \
  --parallelism 2 \
  --fake-kimi training/tests/fixtures/fake_kimi.py \
  --output-dir /tmp/lkjai-kimi-fake \
  --max-calls 2
```

## Real Sample

```bash
python3 scripts/kimi_corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --target-tokens 50000 \
  --mode mixed \
  --parallelism 2 \
  --sample-first
```

## Score

```bash
python3 scripts/kimi_corpus/score_corpus.py data/kimi_synthetic/samples \
  --markdown runs/kimi_corpus/sample_score.md \
  --output runs/kimi_corpus/sample_score.json
```

## Verify

```bash
docker compose --progress quiet --profile verify up --build --abort-on-container-exit verify
```

## Progress

```bash
python3 scripts/kimi_corpus/score_corpus.py \
  --manifest training/corpus/kimi-synthetic-v1/manifest.jsonl \
  --summary-only
```

## Stop

```bash
bash scripts/kimi_corpus/stop_background.sh --run-dir runs/kimi_corpus
```
