#!/usr/bin/env python3
"""Deterministic quality checks for Kimi synthetic corpus JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kimi_lib.records import approx_tokens, record_text
from kimi_lib.score import load_tokenizer, score_paths
from kimi_lib.score_extra import compact_summary, markdown_report, summarize_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Kimi synthetic corpus JSONL files.")
    parser.add_argument("paths", nargs="*", help="JSONL files or directories to score.")
    parser.add_argument("--manifest", default="", help="Optional manifest.jsonl to summarize.")
    parser.add_argument("--output", default="", help="Write JSON summary here.")
    parser.add_argument("--markdown", default="", help="Write markdown report here.")
    parser.add_argument("--tokenizer-json", default="", help="Optional tokenizer.json.")
    parser.add_argument("--summary-only", action="store_true", help="Print compact JSON summary.")
    args = parser.parse_args()
    tokenizer = load_tokenizer(Path(args.tokenizer_json)) if args.tokenizer_json else None
    summary = summarize_manifest(Path(args.manifest)) if args.manifest else score_paths([Path(p) for p in args.paths] or [Path("data/kimi_synthetic")], tokenizer)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(compact_summary(summary) if args.summary_only else summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
