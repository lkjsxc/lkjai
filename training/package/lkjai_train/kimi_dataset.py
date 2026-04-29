import json
import os
from collections import Counter
from pathlib import Path

from .corpus import dedupe_rows
from .formatting import row_text
from .kimi_corpus import generate_kimi_corpus, kimi_source_metadata
from .kimi_validate_rows import validate_kimi_row
from .rows import signature


def prepare_kimi_corpus(paths, target_tokens: int = 60_000_000, seed: int = 42) -> Path:
    paths.ensure()
    target_tokens = int(os.environ.get("TRAIN_FIRST_PARTY_SFT_TOKENS", str(target_tokens)))
    target = Path(os.environ.get("KIMI_OUTPUT_DIR", str(paths.kimi_corpus)))
    row_target = _estimate_rows_for_tokens(target_tokens)
    rows = generate_kimi_corpus(row_target, seed)
    split = {"train": [], "val": [], "holdout": []}
    for row in rows:
        split[row["meta"]["split"]].append(row)
    write_chunked_rows(target / "train", "train", split["train"])
    write_chunked_rows(target / "val", "val", split["val"])
    write_chunked_rows(target / "holdout", "holdout", split["holdout"])
    manifest = _build_manifest(target, rows, split)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def _estimate_rows_for_tokens(target_tokens: int) -> int:
    rough_tokens_per_row = int(os.environ.get("KIMI_TOKENS_PER_ROW", "2000"))
    return max(1000, target_tokens // rough_tokens_per_row)

def _build_manifest(target: Path, rows: list[dict], split: dict) -> dict:
    unique_rows = {signature(row) for row in rows}
    return {
        "schema": "lkjai-agent-jsonl-v2",
        "corpus": "kimi-corpus",
        "rows": len(rows),
        "path": str(target),
        "split_rows": {k: len(v) for k, v in split.items()},
        "unique_rows": len(unique_rows),
        "duplicate_rows": len(rows) - len(unique_rows),
        "sources": kimi_source_metadata(rows),
    }


def validate_kimi_corpus(paths) -> Path:
    report = validate_dataset_directory(paths.kimi_corpus)
    enforce_kimi_report(report)
    paths.kimi_validation_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return paths.kimi_validation_report

def validate_dataset_directory(directory: Path) -> dict:
    files = sorted(directory.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"no .jsonl files found in {directory}")
    total_rows = 0
    signatures = set()
    split_counts = Counter()
    tool_counts = Counter()
    mode_counts = Counter()
    flag_counts = Counter()
    provenance_counts = Counter()
    source_license_counts = Counter()
    trace_finish = 0
    everyday = 0
    everyday_generic = 0
    generic = 0
    xml_valid = 0
    xml_total = 0
    chunk_sizes = {}
    for file in files:
        chunk_sizes[str(file.relative_to(directory))] = len(file.read_text(encoding="utf-8").splitlines())
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            signatures.add(_row_signature(row))
            facts = validate_kimi_row(row)
            meta = row.get("meta", row.get("metadata", {}))
            tags = set(row.get("tags", []))
            is_everyday = "everyday_chat" in tags
            everyday += int(is_everyday)
            split_counts[facts.split] += 1
            mode_counts[facts.mode] += 1
            provenance_counts[meta.get("provenance", "unknown")] += 1
            source_license_counts[(facts.domain, facts.license)] += 1
            xml_total += facts.xml_total
            xml_valid += facts.xml_valid
            if facts.final_tool:
                tool_counts[facts.final_tool] += 1
            trace_finish += int(facts.mode == "sft" and facts.final_tool == "agent.finish")
            if final_action := _final_action(row):
                generic_final = _is_generic_final(final_action)
                generic += int(generic_final)
                everyday_generic += int(is_everyday and generic_final)
            for flag in facts.flags:
                flag_counts[flag] += 1
    duplicate_rate = (total_rows - len(signatures)) / max(1, total_rows)
    token_count = _count_tokens_in_directory(directory)
    return {
        "total_rows": total_rows,
        "split_rows": dict(split_counts),
        "tokenizer_tokens": token_count,
        "duplicate_rate": duplicate_rate,
        "xml_validity_rate": xml_valid / max(1, xml_total),
        "agent_finish_rate": trace_finish / max(1, mode_counts["sft"]),
        "tool_distribution": dict(tool_counts),
        "mode_distribution": dict(mode_counts),
        "flag_counts": dict(flag_counts),
        "chunk_sizes": chunk_sizes,
        "everyday_chat_rows": everyday,
        "everyday_chat_generic_finals": everyday_generic,
        "generic_final_rate": generic / max(1, total_rows),
        "provenance_distribution": dict(provenance_counts),
        "source_license_distribution": [{"domain": d, "license": l, "rows": c} for (d, l), c in sorted(source_license_counts.items())],
    }


def _is_generic_final(action: dict) -> bool:
    text = str(action.get("content", "")).lower()
    generic = ("completed task for", "done.", "ok.", "it depends")
    return action.get("tool") == "agent.finish" and any(item in text for item in generic)

def _row_signature(row: dict) -> str:
    if row.get("mode") == "pretrain":
        return json.dumps({"id": row.get("id"), "text": row.get("text", "")}, sort_keys=True)
    return signature(row)

def _final_action(row: dict) -> dict:
    try:
        from .dataset import parse_assistant_xml

        assistants = [m for m in row.get("messages", []) if m.get("role") == "assistant"]
        return parse_assistant_xml(assistants[-1]["content"]) if assistants else {}
    except (ValueError, KeyError, TypeError):
        return {}

def enforce_kimi_report(report: dict) -> None:
    if report["xml_validity_rate"] < 0.995:
        raise ValueError("Kimi corpus XML validity below 0.995")
    if report["agent_finish_rate"] < 1.0:
        raise ValueError("Kimi corpus traces must all end with agent.finish")
    if report["duplicate_rate"] > 0.01:
        raise ValueError("Kimi corpus duplicate rate exceeds 1%")
    if report["generic_final_rate"] > 0.005:
        raise ValueError("Kimi corpus generic final-answer rate exceeds 0.5%")
    if report["everyday_chat_generic_finals"] != 0:
        raise ValueError("Kimi corpus everyday chat contains generic final answers")
    if report["everyday_chat_rows"] == 0:
        raise ValueError("Kimi corpus missing everyday_chat rows")
    bad_chunks = [path for path, size in report["chunk_sizes"].items() if size > 1000]
    if bad_chunks:
        raise ValueError(f"Kimi corpus chunks exceed 1000 rows: {bad_chunks}")
    if set(report["provenance_distribution"]) != {"kimi-generated"}:
        raise ValueError("Kimi corpus must use only kimi-generated provenance")
    if set(report.get("mode_distribution", {})) - {"pretrain", "sft"}:
        raise ValueError("Kimi corpus contains unknown row modes")
    if report.get("flag_counts"):
        raise ValueError(f"Kimi corpus has invalid rows: {sorted(report['flag_counts'])[:8]}")

def _count_tokens_in_directory(directory: Path) -> int:
    from .formatting import load_rows
    files = sorted(directory.rglob("*.jsonl"))
    texts = []
    for file in files:
        for row in load_rows(file):
            texts.append(row_text(row))
    if not texts:
        return 0
    try:
        from .tokenizer import train_text_tokenizer
    except Exception:
        return sum(len(t) // 4 for t in texts)
    tokenizer = train_text_tokenizer(texts, 8192)
    return sum(len(tokenizer.encode(text).ids) for text in texts)

def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_chunked_rows(directory: Path, split: str, rows: list[dict], chunk_size: int = 1000) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.jsonl"):
        old.unlink()
    for index in range(0, len(rows), chunk_size):
        chunk = rows[index : index + chunk_size]
        path = directory / f"{split}-{index // chunk_size + 1:06d}.jsonl"
        write_rows(path, chunk)
