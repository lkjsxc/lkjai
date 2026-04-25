from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree


def normalized_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")).hexdigest()


def simhash(text: str) -> int:
    grams = char_ngrams(re.sub(r"\s+", " ", text.lower()), 5)
    if not grams:
        return 0
    weights = [0] * 64
    for gram in grams:
        value = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight > 0)


def estimate_near_duplicates(buckets: dict[int, list[int]]) -> int:
    count = 0
    for values in buckets.values():
        seen = []
        for value in values[:2000]:
            if any((value ^ other).bit_count() <= 3 for other in seen):
                count += 1
            seen.append(value)
    return count


def char_ngrams(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 0.0 if len(lines) < 4 else 1.0 - len(set(lines)) / len(lines)


def repeated_ngram_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n * 3:
        return 0.0
    grams = list(zip(*(words[i:] for i in range(n))))
    return 1.0 - len(set(grams)) / max(1, len(grams))


def language_matches(text: str, language: str) -> bool:
    ja = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
    ratio = ja / max(1, len(text))
    return ratio > 0.08 if language == "ja" else ratio < 0.05 if language == "en" else ratio > 0.02


def iter_jsonl_files(paths) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.exists():
            files.append(path)
    return sorted({file for file in files if file.name != "manifest.jsonl"})


def valid_xml(content: str) -> bool:
    if "<action " in content:
        return False
    try:
        root = ElementTree.fromstring(content.strip())
    except ElementTree.ParseError:
        return False
    return root.tag == "action" and not root.attrib and any(child.tag == "tool" and (child.text or "") for child in root)


def summarize_manifest(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if row.get("validation_status") == "valid"]
    elapsed = manifest_elapsed_hours(rows)
    approx = sum(int(row.get("approx_tokens", 0)) for row in valid)
    return {
        "schema": "lkjai-kimi-manifest-summary-v1",
        "manifest": str(path),
        "shards": len(rows),
        "valid_shards": len(valid),
        "quarantined_shards": sum(1 for row in rows if row.get("validation_status") == "quarantined"),
        "failed_shards": sum(1 for row in rows if row.get("validation_status") == "failed"),
        "approx_tokens": approx,
        "tokenizer_tokens": sum(int(row.get("tokenizer_tokens", 0)) for row in valid),
        "generated_documents": sum(int(row.get("generated_documents", 0)) for row in valid),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
        "failure_count": sum(1 for row in rows if row.get("validation_status") != "valid"),
        "tokens_per_hour": round(approx / elapsed, 2) if elapsed else 0.0,
        "kimi_calls_per_hour": round(len(rows) / elapsed, 2) if elapsed else 0.0,
    }


def manifest_elapsed_hours(rows: list[dict]) -> float:
    stamps = []
    for row in rows:
        try:
            stamps.append(datetime.fromisoformat(str(row.get("created_at", "")).replace("Z", "+00:00")).timestamp())
        except ValueError:
            pass
    return 0.0 if len(stamps) < 2 else max(1e-9, (max(stamps) - min(stamps)) / 3600)


def compact_summary(summary: dict) -> dict:
    keys = ["documents", "valid_documents", "approx_tokens", "pretraining_approx_tokens", "sft_supervised_approx_tokens", "tokenizer_tokens", "duplicate_rate", "near_duplicate_rate", "mean_score", "flag_counts", "shards", "valid_shards", "quarantined_shards", "failed_shards", "tokens_per_hour", "kimi_calls_per_hour"]
    return {key: summary.get(key) for key in keys if key in summary}


def markdown_report(summary: dict) -> str:
    lines = ["# Kimi Corpus Score", ""]
    for key, value in compact_summary(summary).items():
        lines.append(f"- `{key}`: `{value}`")
    for key in ["mode_distribution", "language_distribution", "domain_distribution"]:
        lines.extend(["", f"## {key}", ""])
        for name, count in sorted(summary.get(key, {}).items()):
            lines.append(f"- {name}: {count}")
    return "\n".join(lines) + "\n"
