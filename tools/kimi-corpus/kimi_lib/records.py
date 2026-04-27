from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_jsonl_payload(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```") or not stripped.startswith("{"):
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return rows


def normalize_record(record: dict, mode: str, index: int, prompt_version: str, split: str) -> dict:
    if mode == "pretrain":
        record.setdefault("id", f"kimi-pretrain-{int(time.time())}-{index:04d}")
        record["mode"] = "pretrain"
        for key, value in {"language": "en", "domain": "general", "difficulty": "intermediate"}.items():
            record.setdefault(key, value)
        record.setdefault("title", record["id"])
        record.setdefault("text", "")
        metadata = record.setdefault("metadata", {})
        metadata.setdefault("source", "kimi_synthetic")
        metadata.setdefault("mode", "pretrain")
        metadata.setdefault("generated_at", now_iso())
        metadata.setdefault("prompt_version", prompt_version)
        metadata.setdefault("estimated_tokens", approx_tokens(record.get("text", "")))
        metadata.setdefault("provenance", "kimi-generated")
        metadata.setdefault("author_type", "external-agent-generated")
        metadata.setdefault("author_model", "kimi-code")
        metadata.setdefault("language", "en")
        metadata.setdefault("license", "project-local")
        metadata.setdefault("source_ref", f"kimi_synthetic:{prompt_version}")
        return record
    record.setdefault("messages", [])
    record.setdefault("tags", ["kimi_synthetic", "language:en"])
    meta = record.setdefault("meta", {})
    defaults = {
        "id": f"kimi-sft-{int(time.time())}-{index:04d}",
        "split": split,
        "provenance": "kimi-generated",
        "author_type": "external-agent-generated",
        "author_model": "kimi-code",
        "quality_tier": "high",
        "domain": "synthetic-sft",
        "skill": "instruction-following",
        "toolset": "none",
        "language": "en",
        "safety_scope": "workspace-safe",
        "license": "project-local",
        "source_ref": f"kimi_synthetic:{prompt_version}",
        "mode": "sft",
        "prompt_version": prompt_version,
    }
    for key, value in defaults.items():
        meta.setdefault(key, value)
    return record


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def record_text(row: dict) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("messages"), list):
        return "\n".join(str(msg.get("content", "")) for msg in row["messages"])
    return ""


def sample_excerpts(paths: list[Path], limit: int = 8) -> str:
    lines = ["## Representative Excerpts", ""]
    count = 0
    for path in paths:
        if count >= limit or not path.exists():
            break
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if count >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            excerpt = re.sub(r"\s+", " ", record_text(row))[:500]
            lines.append(f"- `{path}:{line_no}` {excerpt}")
            count += 1
    lines.append("")
    return "\n".join(lines)
