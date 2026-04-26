# Kimi Full Corpus V1

This directory is the committed target location for Kimi-generated XML action
corpus chunks.

Expected layout:

```text
manifest.json
validation-report.json
train/train-000001.jsonl
val/val-000001.jsonl
holdout/holdout-000001.jsonl
```

Generate chunks with:

```sh
KIMI_OUTPUT_DIR=corpus/generated/kimi-full-v1 \
PYTHONPATH=training/package \
python3 -m lkjai_train.cli --data-dir data/train prepare-corpus
```

Use about 1000 JSONL rows per chunk. Every assistant message must be one XML
`<action>` and every successful trace must end with `agent.finish`.
