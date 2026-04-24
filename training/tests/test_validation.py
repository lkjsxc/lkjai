import json
from pathlib import Path

import pytest

from lkjai_train.dataset import (
    ALLOWED_PROVENANCE,
    DISALLOWED_AUTHOR_PARTS,
    REQUIRED_META,
    parse_assistant_xml,
    validate_assistant_xml,
    validate_dataset,
    validate_provenance,
)
from lkjai_train.rows import direct_row, meta, row


def test_validate_provenance_rejects_disallowed_author_model():
    for bad in ("gpt-4", "codex", "claude-3", "llama-llm"):
        with pytest.raises(ValueError, match="disallowed"):
            validate_provenance({"author_model": bad, "author_type": "repo-derived", "provenance": "repo-derived"})


def test_validate_provenance_rejects_llm_curated_type():
    with pytest.raises(ValueError, match="disallowed"):
        validate_provenance({"author_model": "none", "author_type": "llm-curated", "provenance": "repo-derived"})


def test_validate_provenance_rejects_bad_provenance():
    with pytest.raises(ValueError, match="disallowed provenance"):
        validate_provenance({"author_model": "none", "author_type": "repo-derived", "provenance": "gpt-generated"})


def test_validate_provenance_accepts_allowed():
    for prov in ALLOWED_PROVENANCE:
        validate_provenance({"author_model": "none", "author_type": prov, "provenance": prov})


def test_validate_assistant_xml_rejects_non_xml():
    with pytest.raises(ValueError, match="not valid XML"):
        validate_assistant_xml("not xml")


def test_validate_assistant_xml_rejects_attributes():
    with pytest.raises(ValueError, match="must not use attributes"):
        validate_assistant_xml('<action kind="tool"></action>')


def test_validate_assistant_xml_rejects_missing_tool():
    with pytest.raises(ValueError, match="missing tool"):
        validate_assistant_xml("<action><content>ok</content></action>")


def test_validate_assistant_xml_accepts_valid():
    validate_assistant_xml("<action><tool>agent.finish</tool><content>ok</content></action>")
    data = parse_assistant_xml("<action><tool>fs.read</tool><path>README.md</path></action>")
    assert data["tool"] == "fs.read"


def test_validate_dataset_rejects_missing_meta(tmp_path):
    bad = row([{"role": "user", "content": "hello"}], [], {"id": "bad"})
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required meta"):
        validate_dataset(path)


def test_validate_dataset_rejects_disallowed_provenance(tmp_path):
    bad = direct_row("hello", "hi", [], meta("bad", "test", "test", "test", split="train"))
    bad["meta"]["author_model"] = "gpt-4"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disallowed"):
        validate_dataset(path)


def test_validate_dataset_rejects_invalid_assistant_xml(tmp_path):
    bad = row(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "not json"}],
        [],
        meta("bad", "test", "test", "test", split="train"),
    )
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid XML"):
        validate_dataset(path)


def test_validate_dataset_rejects_high_duplicate_rate(tmp_path):
    good = direct_row("hello", "hi", ["test"], meta("good", "test", "test", "test", split="train"))
    lines = [json.dumps(good) for _ in range(200)]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate rate exceeds"):
        validate_dataset(path)


def test_assistant_content_is_valid_xml():
    from lkjai_train.corpus import generate_corpus

    rows = generate_corpus(1000)
    for row in rows:
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                data = parse_assistant_xml(message["content"])
                assert "tool" in data


def test_duplicate_rate_below_threshold():
    from lkjai_train.corpus import generate_corpus
    from lkjai_train.rows import signature

    rows = generate_corpus(60000)
    sigs = {signature(r) for r in rows}
    assert len(rows) - len(sigs) <= len(rows) * 0.01
