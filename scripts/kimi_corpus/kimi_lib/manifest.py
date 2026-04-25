from __future__ import annotations

import json
import threading
from pathlib import Path

from .records import now_iso


class Manifest:
    def __init__(self, output_dir: Path, kimi_variant: str):
        self.output_dir = output_dir
        self.path = output_dir / "manifest.jsonl"
        self.kimi_variant = kimi_variant
        self.lock = threading.Lock()
        self.next_ids = self._next_ids()

    def _next_ids(self) -> dict[str, int]:
        ids = {"pretrain": 1, "sft": 1}
        for row in self.rows():
            mode = row.get("mode")
            if mode in ids:
                ids[mode] = max(ids[mode], int(row.get("shard_id", 0)) + 1)
        return ids

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def reserve(self, mode: str) -> int:
        with self.lock:
            shard_id = self.next_ids[mode]
            self.next_ids[mode] += 1
            return shard_id

    def valid_tokens(self, mode: str = "mixed") -> int:
        total = 0
        for row in self.rows():
            if row.get("validation_status") != "valid":
                continue
            if mode in {"pretrain", "sft"} and row.get("mode") != mode:
                continue
            total += int(row.get("tokenizer_tokens") or row.get("approx_tokens") or 0)
        return total

    def mode_counts(self) -> dict[str, int]:
        counts = {"pretrain": 0, "sft": 0}
        for row in self.rows():
            if row.get("mode") in counts:
                counts[row["mode"]] += 1
        return counts

    def append_success(self, path: Path, mode: str, split: str, shard_id: int, records: list[dict], score: dict, status: str, retries: int, result) -> None:
        entry = {
            "shard_id": shard_id,
            "mode": mode,
            "split": split,
            "prompt_version": records[0].get("metadata", records[0].get("meta", {})).get("prompt_version", ""),
            "target_documents": len(records),
            "generated_documents": len(records),
            "approx_tokens": score.get("approx_tokens", 0),
            "tokenizer_tokens": score.get("tokenizer_tokens", 0) or 0,
            "character_count": score.get("char_count", 0),
            "language_distribution": score.get("language_distribution", {}),
            "domain_distribution": score.get("domain_distribution", {}),
            "validation_status": status,
            "created_at": now_iso(),
            "path": str(path),
            "kimi_command_variant": result.command_variant,
            "retry_count": retries,
            "duration_seconds": result.duration_seconds,
            "failure_count": 0 if status == "valid" else 1,
        }
        self.append(entry)

    def append_failure(self, mode: str, split: str, shard_id: int, retries: int, reason: str) -> None:
        self.append({
            "shard_id": shard_id,
            "mode": mode,
            "split": split,
            "prompt_version": "",
            "target_documents": 0,
            "generated_documents": 0,
            "approx_tokens": 0,
            "tokenizer_tokens": 0,
            "character_count": 0,
            "language_distribution": {},
            "domain_distribution": {},
            "validation_status": "failed",
            "failure_reason": reason,
            "created_at": now_iso(),
            "path": "",
            "kimi_command_variant": self.kimi_variant,
            "retry_count": retries,
            "duration_seconds": 0.0,
            "failure_count": 1,
        })

    def append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def split_for_shard(shard_id: int) -> str:
    if shard_id % 20 == 0:
        return "holdout"
    if shard_id % 10 == 0:
        return "val"
    return "train"
