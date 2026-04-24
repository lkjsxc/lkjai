import json
import os
from collections import Counter
from pathlib import Path

from .corpus import dedupe_rows
from .formatting import SPECIAL_TOKENS, row_text
from .kimi_corpus import generate_kimi_corpus, kimi_source_metadata
from .rows import signature


def prepare_kimi_corpus(paths, target_tokens: int = 500_000_000, seed: int = 42) -> Path:
    paths.ensure()
    row_target = _estimate_rows_for_tokens(target_tokens)
    rows = generate_kimi_corpus(row_target, seed)
    split = {"train": [], "val": [], "holdout": []}
    for row in rows:
        split[row["meta"]["split"]].append(row)
    write_rows(paths.kimi_train, split["train"])
    write_rows(paths.kimi_val, split["val"])
    write_rows(paths.kimi_holdout, split["holdout"])
    manifest = _build_manifest(paths, rows, split)
    paths.kimi_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.kimi_corpus


def _estimate_rows_for_tokens(target_tokens: int) -> int:
    rough_tokens_per_row = int(os.environ.get("KIMI_TOKENS_PER_ROW", "2000"))
    return max(1000, target_tokens // rough_tokens_per_row)


def _build_manifest(paths, rows: list[dict], split: dict) -> dict:
    unique_rows = {signature(row) for row in rows}
    return {
        "schema": "lkjai-agent-jsonl-v2",
        "corpus": "kimi-corpus",
        "rows": len(rows),
        "split_rows": {k: len(v) for k, v in split.items()},
        "unique_rows": len(unique_rows),
        "duplicate_rows": len(rows) - len(unique_rows),
        "sources": kimi_source_metadata(rows),
    }


def validate_kimi_corpus(paths) -> Path:
    report = validate_dataset_directory(paths.kimi_corpus)
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
    provenance_counts = Counter()
    source_license_counts = Counter()
    trace_finish = 0
    xml_valid = 0
    xml_total = 0
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            signatures.add(signature(row))
            meta = row.get("meta", {})
            split_counts[meta.get("split", "unknown")] += 1
            provenance_counts[meta.get("provenance", "unknown")] += 1
            source_license_counts[(meta.get("domain", "unknown"), meta.get("license", "unknown"))] += 1
            last_assistant = None
            for message in row.get("messages", []):
                if message.get("role") == "assistant":
                    xml_total += 1
                    last_assistant = message["content"]
                    try:
                        from .dataset import parse_assistant_xml
                        parsed = parse_assistant_xml(message["content"])
                        xml_valid += 1
                        tool_counts[parsed.get("tool", "unknown")] += 1
                    except ValueError:
                        pass
            if last_assistant:
                try:
                    from .dataset import parse_assistant_xml
                    if parse_assistant_xml(last_assistant).get("tool") == "agent.finish":
                        trace_finish += 1
                except ValueError:
                    pass
    duplicate_rate = (total_rows - len(signatures)) / max(1, total_rows)
    token_count = _count_tokens_in_directory(directory)
    return {
        "total_rows": total_rows,
        "split_rows": dict(split_counts),
        "tokenizer_tokens": token_count,
        "duplicate_rate": duplicate_rate,
        "xml_validity_rate": xml_valid / max(1, xml_total),
        "agent_finish_rate": trace_finish / max(1, total_rows),
        "tool_distribution": dict(tool_counts),
        "provenance_distribution": dict(provenance_counts),
        "source_license_distribution": [{"domain": d, "license": l, "rows": c} for (d, l), c in sorted(source_license_counts.items())],
    }


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
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except Exception:
        return sum(len(t) // 4 for t in texts)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return sum(len(tokenizer.encode(text).ids) for text in texts)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
