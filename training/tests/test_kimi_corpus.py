import json
from pathlib import Path

import pytest

from lkjai_train.dataset import parse_assistant_xml
from lkjai_train.kimi_corpus import generate_kimi_corpus
from lkjai_train.kimi_dataset import validate_dataset_directory, write_rows
from lkjai_train.paths import Paths
from lkjai_train.rows import kimi_meta


def test_kimi_meta_has_correct_provenance():
    m = kimi_meta("test-001", "domain", "skill", "ref", split="train")
    assert m["provenance"] == "kimi-generated"
    assert m["author_type"] == "external-agent-generated"
    assert m["author_model"] == "kimi-code"


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
