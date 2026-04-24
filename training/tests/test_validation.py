import json
from pathlib import Path

import pytest

from lkjai_train.dataset import (
    ALLOWED_PROVENANCE,
    DISALLOWED_AUTHOR_PARTS,
    REQUIRED_META,
    validate_assistant_json,
    validate_dataset,
    validate_provenance,
)
from lkjai_train.rows import direct_row, meta, row


def test_validate_provenance_rejects_disallowed_author_model():
    for bad in ("gpt-4", "kimi", "claude-3", "llama-llm"):
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


def test_validate_assistant_json_rejects_non_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_assistant_json("not json")


def test_validate_assistant_json_rejects_non_object():
    with pytest.raises(ValueError, match="must be an object"):
        validate_assistant_json('["list"]')


def test_validate_assistant_json_rejects_missing_kind():
    with pytest.raises(ValueError, match="missing kind"):
        validate_assistant_json('{"tool":"fs.read"}')


def test_validate_assistant_json_accepts_valid():
    validate_assistant_json('{"kind":"final","content":"ok"}')
    validate_assistant_json('{"kind":"tool_call","tool":"fs.read","args":{}}')


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


def test_validate_dataset_rejects_invalid_assistant_json(tmp_path):
    bad = row(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "not json"}],
        [],
        meta("bad", "test", "test", "test", split="train"),
    )
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_dataset(path)


def test_validate_dataset_rejects_high_duplicate_rate(tmp_path):
    good = direct_row("hello", "hi", ["test"], meta("good", "test", "test", "test", split="train"))
    lines = [json.dumps(good) for _ in range(200)]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate rate exceeds"):
        validate_dataset(path)
