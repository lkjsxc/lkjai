#!/usr/bin/env python3
"""Build compact Kimi sample reports and optional prompt-refinement requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kimi_lib.kimi_cli import KimiRunner
from kimi_lib.records import record_text
from kimi_lib.score import score_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Kimi sample shards.")
    parser.add_argument("--samples", default="data/kimi_synthetic/samples")
    parser.add_argument("--output", default="runs/kimi_corpus/sample_report.md")
    parser.add_argument("--refine-with-kimi", action="store_true")
    parser.add_argument("--prompt-refiner", default="scripts/kimi_corpus/prompts/prompt_refiner.txt")
    args = parser.parse_args()
    samples, output = Path(args.samples), Path(args.output)
    summary = score_paths([samples])
    report = markdown(summary, representative_excerpts(samples))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    if args.refine_with_kimi:
        refine_prompts(report, Path(args.prompt_refiner))
    print(json.dumps({"report": str(output), "documents": summary.get("documents", 0), "mean_score": summary.get("mean_score", 0)}))


def representative_excerpts(samples: Path, limit: int = 8) -> list[dict]:
    excerpts = []
    for path in sorted(samples.rglob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if len(excerpts) >= limit:
                return excerpts
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            excerpts.append({"path": str(path), "line": line_no, "excerpt": record_text(row)[:500]})
    return excerpts


def markdown(summary: dict, excerpts: list[dict]) -> str:
    lines = ["# Kimi Sample Inspection", ""]
    for key in ["documents", "valid_documents", "approx_tokens", "duplicate_rate", "near_duplicate_rate", "mean_score", "flag_counts"]:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    lines.extend(["", "## Representative Excerpts", ""])
    for item in excerpts:
        lines.append(f"- `{item['path']}:{item['line']}` {item['excerpt'].replace(chr(10), ' ')[:500]}")
    return "\n".join(lines) + "\n"


def refine_prompts(report: str, refiner_path: Path) -> None:
    prompt = refiner_path.read_text(encoding="utf-8").replace("{{SAMPLE_SUMMARY}}", report[:8000])
    result = KimiRunner(Path("runs/kimi_corpus/logs")).invoke(prompt, "prompt_refiner", 240, 0)
    if result.returncode != 0:
        raise RuntimeError(f"kimi prompt refinement failed: logs={result.stderr_path}")
    Path("scripts/kimi_corpus/prompts/prompt_refiner_candidate.txt").write_text(result.stdout_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


if __name__ == "__main__":
    main()
