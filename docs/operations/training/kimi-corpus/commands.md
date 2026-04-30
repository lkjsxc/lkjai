# Commands

## Setup

```bash
kimi --help
kimi info
python3 tools/kimi-corpus/generate_kimi_corpus.py --help
python3 tools/kimi-corpus/score_corpus.py --help
```

## Fake Smoke

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --target-tokens 50000 \
  --mode mixed \
  --parallelism 2 \
  --output-dir /tmp/lkjai-kimi-fake \
  --max-calls 2
```

## Real Sample

```bash
python3 tools/kimi-corpus/generate_kimi_corpus.py \
  --config configs/corpus/kimi_debug.yaml \
  --target-tokens 50000 \
  --mode mixed \
  --parallelism 2 \
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

## Stop

```bash
bash tools/kimi-corpus/stop_background.sh --run-dir runs/kimi_corpus
```
