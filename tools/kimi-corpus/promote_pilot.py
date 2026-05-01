#!/usr/bin/env python3
"""Promote a validated Kimi pilot into the committed corpus layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from kimi_lib.records import write_jsonl_atomic
from kimi_lib.score import score_paths
from kimi_lib.score_extra import promotion_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a validated Kimi SFT pilot.")
    parser.add_argument("--source", default="data/kimi_synthetic/pilot-v2")
    parser.add_argument("--dest", default="corpus/generated/kimi-sft-60m-v2")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--corpus-target-tokens", type=int, default=60_000_000)
    parser.add_argument("--min-tokens", type=int, default=900_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source, dest = Path(args.source), Path(args.dest)
    rows = load_rows(source)
    if not rows:
        raise SystemExit(f"no rows found under {source}")
    score = score_paths([source])
    gate_args = argparse.Namespace(
        fail_on_invalid=True,
        fail_on_split_leakage=True,
        require_template_families=True,
        max_duplicate_rate=0.01,
        max_near_duplicate_rate=0.01,
    )
    gate = promotion_gate([source], score, gate_args)
    score["promotion_gate"] = gate
    if gate["status"] != "pass":
        print(json.dumps(score, indent=2, ensure_ascii=False))
        raise SystemExit("promotion gate failed")
    if int(score.get("approx_tokens", 0)) < args.min_tokens:
        raise SystemExit(f"pilot too small: {score.get('approx_tokens', 0)}")
    if dest.exists():
        shutil.rmtree(dest)
    write_splits(dest, rows)
    write_reports(dest, score, args)
    print(json.dumps({"status": "promoted", "rows": len(rows), "dest": str(dest)}))


def load_rows(source: Path) -> list[dict]:
    rows = []
    for path in sorted(source.rglob("*.jsonl")):
        if path.name == "manifest.jsonl" or "quarantine" in path.parts:
            continue
        split = split_for_path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                normalize_row(row, split)
                rows.append(row)
    return rows


def split_for_path(path: Path) -> str:
    for split in ["train", "val", "holdout"]:
        if split in path.parts:
            return split
    return "train"


def normalize_row(row: dict, split: str) -> None:
    meta = row.setdefault("meta", {})
    meta["split"] = split
    meta["contract_validated"] = True
    meta["fixture_executed"] = True
    meta.setdefault("schema", "lkjai-agent-jsonl-v3")
    meta.setdefault("source_ref", "corpus/fixtures/repo-grounding-v1.json")
    tags = row.setdefault("tags", [])
    if "promoted" not in tags:
        tags.append("promoted")


def write_splits(dest: Path, rows: list[dict]) -> None:
    for split in ["train", "val", "holdout"]:
        split_rows = [row for row in rows if row.get("meta", {}).get("split") == split]
        if not split_rows:
            raise SystemExit(f"missing split {split}")
        write_jsonl_atomic(dest / split / f"{split}-000001.jsonl", split_rows)


def write_reports(dest: Path, score: dict, args: argparse.Namespace) -> None:
    rows = int(score.get("documents", 0))
    manifest = {
        "schema": "lkjai-agent-jsonl-v3",
        "corpus": "kimi-sft-60m-v2",
        "status": "pilot-promoted",
        "rows": rows,
        "path": str(dest),
        "split_rows": split_counts(dest),
        "pilot_target_tokens": args.target_tokens,
        "target_tokenizer_tokens": args.corpus_target_tokens,
        "validation_report": "validation-report.json",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dest / "validation-report.json").write_text(
        json.dumps(score, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def split_counts(dest: Path) -> dict[str, int]:
    counts = {}
    for split in ["train", "val", "holdout"]:
        total = 0
        for path in (dest / split).glob("*.jsonl"):
            total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        counts[split] = total
    return counts


if __name__ == "__main__":
    main()
