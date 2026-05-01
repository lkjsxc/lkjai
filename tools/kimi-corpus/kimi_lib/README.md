# Kimi Library

## Purpose

Python modules under this directory implement Kimi API generation, scoring,
normalization, and manifest handling.

## Contents

- [__init__.py](__init__.py): package marker.
- [config.py](config.py): config loading and CLI overrides.
- [generator.py](generator.py): shard generation and sample workflow.
- [kimi_api.py](kimi_api.py): Moonshot-compatible HTTP client.
- [kimi_cli.py](kimi_cli.py): legacy CLI runner and result shape.
- [kimi_keys.py](kimi_keys.py): API key loading, fingerprinting, and redaction.
- [manifest.py](manifest.py): shard manifest accounting.
- [prompts.py](prompts.py): prompt template rendering.
- [records.py](records.py): JSONL parsing and row normalization.
- [sample_report.py](sample_report.py): sample-report formatting.
- [score.py](score.py): corpus scoring entrypoint.
- [score_agent.py](score_agent.py): XML-action SFT validation.
- [score_extra.py](score_extra.py): dedup, language, and report helpers.

## Rules

- Never log raw API keys.
- Stage generated rows under ignored `data/` or `runs/` until promoted.
