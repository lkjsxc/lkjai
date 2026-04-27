from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from .records import approx_tokens, record_text
from .score_extra import (
    estimate_near_duplicates,
    iter_jsonl_files,
    language_matches,
    normalized_hash,
    repeated_line_ratio,
    repeated_ngram_ratio,
    simhash,
    valid_xml,
)


CHAT = ("user:", "assistant:", "as an ai", "i cannot", "i'm unable", "chatgpt", "kimi cli", "usage: kimi")
BOILER = ("here is", "strict jsonl", "as requested", "i hope this helps")
UNSAFE = ("make a bomb", "steal password", "credit card dump", "malware payload")
PRETRAIN = {"id", "mode", "language", "domain", "difficulty", "title", "text", "metadata"}
PRETRAIN_META = {"source", "mode", "generated_at", "prompt_version", "estimated_tokens", "provenance", "author_type", "author_model", "language", "license", "source_ref"}
SFT = {"messages", "tags", "meta"}
SFT_META = {"id", "split", "provenance", "author_type", "author_model", "quality_tier", "domain", "skill", "toolset", "language", "safety_scope", "license", "source_ref", "mode", "prompt_version"}


@dataclass
class DocumentScore:
    path: str
    line: int
    doc_id: str
    mode: str
    valid: bool
    score: float
    approx_tokens: int
    tokenizer_tokens: int | None
    char_count: int
    flags: list[str]


def score_paths(paths: list[Path], tokenizer=None) -> dict:
    docs, bad_json, hashes, buckets = [], [], Counter(), defaultdict(list)
    languages, domains, modes = Counter(), Counter(), Counter()
    chars = pretrain_tokens = sft_tokens = sft_loss = 0
    for path in iter_jsonl_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    bad_json.append({"path": str(path), "line": line_no, "error": str(error)})
                    continue
                score = score_record(row, path, line_no, tokenizer)
                text = record_text(row)
                docs.append(score); hashes[normalized_hash(text)] += 1
                buckets[simhash(text) >> 48].append(simhash(text))
                languages[record_language(row)] += 1; domains[record_domain(row)] += 1
                modes[score.mode] += 1; chars += score.char_count
                if score.mode == "pretrain":
                    pretrain_tokens += score.approx_tokens
                if score.mode == "sft":
                    sft_tokens += score.approx_tokens; sft_loss += approx_tokens(assistant_text(row))
    dupes = sum(count - 1 for count in hashes.values() if count > 1)
    flags = Counter(flag for item in docs for flag in item.flags)
    token_sum = sum(item.approx_tokens for item in docs)
    exact = sum(item.tokenizer_tokens or 0 for item in docs) if any(item.tokenizer_tokens is not None for item in docs) else None
    return {
        "schema": "lkjai-kimi-score-v1",
        "files": [str(path) for path in iter_jsonl_files(paths)],
        "documents": len(docs),
        "valid_documents": sum(1 for item in docs if item.valid),
        "invalid_json_lines": bad_json,
        "approx_tokens": token_sum,
        "pretraining_approx_tokens": pretrain_tokens,
        "sft_approx_tokens": sft_tokens,
        "sft_supervised_approx_tokens": sft_loss,
        "tokenizer_tokens": exact,
        "char_count": chars,
        "duplicate_documents": dupes,
        "duplicate_rate": dupes / max(1, len(docs)),
        "near_duplicate_documents": estimate_near_duplicates(buckets),
        "near_duplicate_rate": estimate_near_duplicates(buckets) / max(1, len(docs)),
        "language_distribution": dict(languages),
        "domain_distribution": dict(domains),
        "mode_distribution": dict(modes),
        "flag_counts": dict(flags),
        "mean_score": sum(item.score for item in docs) / max(1, len(docs)),
        "low_quality_documents": sum(1 for item in docs if item.score < 0.7),
        "documents_detail": [asdict(item) for item in docs[:1000]],
    }


def score_record(row: dict, path: Path, line_no: int, tokenizer=None) -> DocumentScore:
    mode = str(row.get("mode") or row.get("meta", {}).get("mode") or row.get("metadata", {}).get("mode") or infer_mode(row))
    text, flags = record_text(row), []
    flags.extend(validate_pretrain(row) if mode == "pretrain" else validate_sft(row) if mode == "sft" else ["unknown_mode"])
    flags.extend(content_flags(text, mode))
    lang = record_language(row)
    if lang != "en":
        flags.append("invalid_language")
    elif not language_matches(text, lang):
        flags.append("language_mismatch")
    exact = tokenizer_token_count(tokenizer, text) if tokenizer is not None else None
    valid = not flags or all(flag == "language_mismatch" for flag in flags)
    return DocumentScore(str(path), line_no, str(row.get("id") or row.get("meta", {}).get("id") or ""), mode, valid, round(max(0.0, 1.0 - min(0.95, 0.08 * len(set(flags)))), 4), approx_tokens(text), exact, len(text), sorted(set(flags)))


def validate_pretrain(row: dict) -> list[str]:
    flags = []
    if PRETRAIN - row.keys(): flags.append("missing_pretrain_fields")
    if row.get("mode") != "pretrain": flags.append("wrong_pretrain_mode")
    if not isinstance(row.get("text"), str) or len(row.get("text", "").strip()) < 80: flags.append("short_or_empty_text")
    meta = row.get("metadata", {})
    if not isinstance(meta, dict) or meta.get("source") != "kimi_synthetic": flags.append("bad_metadata")
    elif PRETRAIN_META - meta.keys(): flags.append("missing_pretrain_meta_fields")
    elif meta.get("provenance") != "kimi-generated" or meta.get("language") != "en": flags.append("bad_pretrain_meta")
    return flags


def validate_sft(row: dict) -> list[str]:
    flags, messages, meta = [], row.get("messages", []), row.get("meta", {})
    if SFT - row.keys(): flags.append("missing_sft_fields")
    if not isinstance(messages, list) or len(messages) < 2: flags.append("bad_messages")
    elif not any(msg.get("role") == "assistant" for msg in messages): flags.append("missing_assistant")
    if not isinstance(meta, dict) or meta.get("provenance") != "kimi-generated": flags.append("bad_sft_meta")
    elif SFT_META - meta.keys(): flags.append("missing_sft_meta_fields")
    elif meta.get("language") != "en": flags.append("bad_sft_language")
    for message in messages if isinstance(messages, list) else []:
        if message.get("role") == "assistant" and not valid_xml(message.get("content", "")):
            flags.append("invalid_assistant_xml"); break
    return flags


def content_flags(text: str, mode: str) -> list[str]:
    lower, flags = text.lower(), []
    if not text.strip(): return ["empty_text"]
    if "```json" in lower or lower.startswith("[") or lower.startswith("{\"documents\""): flags.append("json_wrapper_or_fence")
    if lower.count("```") % 2 == 1: flags.append("unbalanced_markdown_fence")
    if any(term in lower for term in UNSAFE): flags.append("unsafe_keyword")
    if any(term in lower for term in BOILER): flags.append("boilerplate")
    if mode == "pretrain" and any(term in lower for term in CHAT): flags.append("pretrain_chat_contamination")
    if repeated_line_ratio(text) > 0.25: flags.append("repeated_lines")
    if repeated_ngram_ratio(text) > 0.35: flags.append("repeated_ngrams")
    if re.search(r"(.)\1{18,}", text): flags.append("same_character_run")
    return flags


def record_language(row: dict) -> str:
    value = row.get("language") or row.get("meta", {}).get("language") or row.get("metadata", {}).get("language")
    if value: return str(value)
    for tag in row.get("tags", []) if isinstance(row.get("tags"), list) else []:
        if isinstance(tag, str) and tag.startswith("language:"): return tag.split(":", 1)[1]
    return "unknown"


def record_domain(row: dict) -> str:
    return str(row.get("domain") or row.get("meta", {}).get("domain") or row.get("metadata", {}).get("domain") or "unknown")


def assistant_text(row: dict) -> str:
    return "\n".join(str(msg.get("content", "")) for msg in row.get("messages", []) if msg.get("role") == "assistant")


def infer_mode(row: dict) -> str:
    return "pretrain" if "text" in row else "sft" if "messages" in row else "unknown"


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(path)) if path.exists() else None
    except Exception:
        return None


def tokenizer_token_count(tokenizer, text: str) -> int | None:
    return None if tokenizer is None else len(tokenizer.encode(text).ids)
