import json
from pathlib import Path

import pytest

from lkjai_train.dataset import parse_assistant_xml
from lkjai_train.kimi_corpus import generate_kimi_corpus
from lkjai_train.kimi_dataset import validate_dataset_directory, write_chunked_rows, write_rows
from lkjai_train.kimi_validate_rows import validate_kimi_row
from lkjai_train.paths import Paths
from lkjai_train.rows import kimi_meta


def test_kimi_meta_has_correct_provenance():
    m = kimi_meta("test-001", "domain", "skill", "ref", split="train")
    assert m["provenance"] == "kimi-generated"
    assert m["author_type"] == "external-agent-generated"
    assert m["author_model"] == "kimi-code"
    assert m["mode"] == "sft"
    assert m["prompt_version"] == "deterministic-v1"


def test_generate_kimi_corpus_produces_xml_actions():
    rows = generate_kimi_corpus(50)
    assert rows
    for row in rows:
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                parsed = parse_assistant_xml(message["content"])
                assert "tool" in parsed


def test_all_traces_end_with_agent_finish():
    rows = generate_kimi_corpus(50)
    for row in rows:
        msgs = row["messages"]
        last_assistant = None
        for msg in msgs:
            if msg.get("role") == "assistant":
                last_assistant = msg["content"]
        assert last_assistant is not None
        assert parse_assistant_xml(last_assistant)["tool"] == "agent.finish"


def test_active_kimi_rows_do_not_teach_rejected_assistant_targets():
    rows = generate_kimi_corpus(200)
    assert not [row for row in rows if "preference" in row.get("tags", [])]


def test_kimi_corpus_contains_everyday_chat():
    rows = generate_kimi_corpus(100)
    everyday = [row for row in rows if "everyday_chat" in row.get("tags", [])]
    assert len(everyday) >= 40
    parsed = parse_assistant_xml(everyday[0]["messages"][-1]["content"])
    assert parsed["tool"] == "agent.finish"
    assert parsed.get("reasoning")


def test_directory_validation_produces_report(tmp_path):
    paths = Paths(str(tmp_path))
    rows = generate_kimi_corpus(20)
    write_rows(paths.kimi_train, [r for r in rows if r["meta"]["split"] == "train"])
    write_rows(paths.kimi_val, [r for r in rows if r["meta"]["split"] == "val"])
    write_rows(paths.kimi_holdout, [r for r in rows if r["meta"]["split"] == "holdout"])
    report = validate_dataset_directory(paths.kimi_corpus)
    assert report["total_rows"] == len(rows)
    assert report["agent_finish_rate"] == 1.0
    assert report["xml_validity_rate"] == 1.0
    assert report["duplicate_rate"] <= 0.01
    assert report["provenance_distribution"]["kimi-generated"] == len(rows)
    assert report["everyday_chat_rows"] > 0
    assert not report["flag_counts"]


def test_validation_rejects_preference_rows_in_active_corpus():
    row = {
        "messages": [
            {"role": "user", "content": "Choose the better answer."},
            {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>bad</content></action>"},
            {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>good</content></action>"},
        ],
        "tags": ["kimi_synthetic", "language:en"],
        "meta": kimi_meta("bad-001", "preferences", "rejected-action", "test", split="train"),
    }
    facts = validate_kimi_row(row)
    assert "preference_row_in_active_corpus" in facts.flags


def test_validation_accepts_english_pretrain_row():
    row = {
        "id": "pretrain-001",
        "mode": "pretrain",
        "language": "en",
        "domain": "science",
        "difficulty": "introductory",
        "title": "Careful Measurement",
        "text": "Careful measurement begins with a clear question, a stable instrument, and a written record of each observation. Repeated trials help separate signal from ordinary noise.",
        "metadata": {
            "source": "kimi_synthetic",
            "mode": "pretrain",
            "generated_at": "2026-04-27T00:00:00Z",
            "prompt_version": "v2",
            "estimated_tokens": 32,
            "provenance": "kimi-generated",
            "author_type": "external-agent-generated",
            "author_model": "kimi-code",
            "language": "en",
            "license": "project-local",
            "source_ref": "kimi_synthetic:v2",
        },
    }
    assert validate_kimi_row(row).valid


def test_chunked_writer_uses_roughly_1000_rows(tmp_path):
    rows = generate_kimi_corpus(1205)
    write_chunked_rows(tmp_path / "train", "train", rows, chunk_size=1000)
    sizes = [len(path.read_text(encoding="utf-8").splitlines()) for path in sorted((tmp_path / "train").glob("*.jsonl"))]
    assert sizes == [1000, 205]
