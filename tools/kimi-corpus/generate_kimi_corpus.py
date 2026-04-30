#!/usr/bin/env python3
"""Resumable Kimi CLI synthetic corpus generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kimi_lib.config import DEFAULT_RUN_DIR, apply_overrides, load_config
from kimi_lib.generator import CorpusGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable Kimi synthetic corpus shards.")
    parser.add_argument("--config", default="configs/corpus/kimi_debug.yaml")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--mode", choices=["pretrain", "sft", "mixed"], default=None)
    parser.add_argument("--pretrain-ratio", type=float, default=None)
    parser.add_argument("--sft-ratio", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--sleep-between-calls", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--batch-documents", type=int, default=None)
    parser.add_argument("--sample-documents", type=int, default=None)
    parser.add_argument("--parallelism", type=int, default=None)
    parser.add_argument("--quarantine-bad-shards", action="store_true")
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--fake-kimi", default="", help="Path to a fake kimi executable for tests.")
    parser.add_argument("--api-provider", choices=["cli", "kimi-api"], default=None)
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--sample-first", action="store_true", help="Run sample, score, refine, second sample workflow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(Path(args.config)), args)
    generator = CorpusGenerator(config, args)
    if args.sample_first:
        generator.sample_first_workflow()
    else:
        generator.run()


if __name__ == "__main__":
    main()
