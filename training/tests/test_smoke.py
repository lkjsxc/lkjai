import json

import pytest

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.corpus import generate_corpus
from lkjai_train.corpus_source import load_entries, tagged_contents, validate_sources
from lkjai_train.dataset import prepare_fixtures, validate_dataset
from lkjai_train.behavioral import action_schema_error, bucket, bucket_rates
from lkjai_train.formatting import prompt_text, supervised_token_ids
from lkjai_train.generation import agent_context_messages, first_json_object, latest_user_event, normalize_action, normalize_messages
from lkjai_train.paths import Paths
from lkjai_train.preference import prepare_preferences
from lkjai_train.public_import import ALLOWED_LICENSES, prepare_public_corpus, validate_public_sources


torch = None
tokenizers = None
try:
    import torch
    import tokenizers
except ImportError:
    pass


class Args:
    command = "smoke"


@pytest.mark.slow
@pytest.mark.skipif(torch is None or tokenizers is None, reason="training deps not installed")
def test_smoke_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_MAX_STEPS", "1")
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert result.name == "fixed-eval.json"
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["pass_rate"] == 1.0
    assert (tmp_path / "checkpoints" / "manifest.json").exists()
    assert (tmp_path / "checkpoints" / "final" / "model.pt").exists()


def test_agent_settings_defaults(monkeypatch):
    monkeypatch.delenv("TRAIN_MODEL_PRESET", raising=False)
    monkeypatch.delenv("TRAIN_BATCH_SIZE", raising=False)
    settings = train_settings("agent")
    assert settings.model_preset == "scratch-60m"
    assert settings.sequence_len == 1024
    assert settings.hidden_size == 640
    assert settings.layers == 12
    assert settings.kv_heads == 2
    assert settings.batch_size == 1
    assert settings.corpus_size == 60000
    assert settings.behavioral_threshold == 0.35


def test_quick_settings_are_tiny():
    settings = train_settings("quick")
    assert settings.sequence_len == 64
    assert settings.hidden_size == 64
    assert settings.max_steps == 5


def test_source_corpus_files_are_tagged_json_arrays():
    validate_sources()
    assert load_entries("general")
    assert tagged_contents("kjxlkj", "search_term")
    assert tagged_contents("public", "public_dataset")


def test_public_sources_are_license_gated(tmp_path):
    paths = Paths(str(tmp_path))
    manifest = validate_public_sources(paths)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["sources"]
    assert {source["license"] for source in data["sources"]}.issubset(ALLOWED_LICENSES)


def test_public_corpus_is_opt_in_without_local_files(tmp_path):
    paths = Paths(str(tmp_path))
    out = prepare_public_corpus(paths)
    assert out.read_text(encoding="utf-8") == ""


def test_fixture_dataset_validates(tmp_path):
    paths = Paths(str(tmp_path))
    fixture = prepare_fixtures(paths)
    assert validate_dataset(fixture) == fixture
    assert validate_dataset(paths.train_dataset) == paths.train_dataset
    assert validate_dataset(paths.val_dataset) == paths.val_dataset
    assert validate_dataset(paths.holdout_dataset) == paths.holdout_dataset


def test_prompt_text_appends_assistant_json_header():
    text = prompt_text([{"role": "user", "content": "hello"}])
    assert text.startswith("<bos>")
    assert text.endswith("<assistant_json>")
    assert "<dialogue>" in text


def test_agent_corpus_default_has_required_mix():
    rows = generate_corpus(60000)
    assert len(rows) == 60000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert "direct_answer" in tags
    assert "docs_grounding" in tags
    assert "runtime_schema" in tags or "tool_trajectory" in tags
    assert tags.count("agentic") >= 10000
    assert tags.count("safety") >= 5000
    assert tags.count("preference") >= 3000
    assert all(row["meta"]["author_model"] == "none" for row in rows)


def test_normalize_action_extracts_first_json_object():
    text = 'noise {"kind":"final","content":"ok"} trailing'
    assert json.loads(normalize_action(text))["content"] == "ok"
    assert first_json_object(text) == '{"kind":"final","content":"ok"}'


def test_normalize_action_returns_raw_invalid_text():
    assert normalize_action("not json <eos>").strip() == "not json"


def test_behavioral_schema_rejects_invalid_action_shape():
    assert action_schema_error({"kind": "33"}).startswith("unknown action kind")
    assert action_schema_error({"kind": "tool_call", "tool": "fs.list", "args": {}}) == ""


def test_behavioral_buckets_report_pass_rates():
    cases = [
        {"bucket": "direct_answer", "passed": True, "valid_json": True},
        {"bucket": "direct_answer", "passed": False, "valid_json": True},
    ]
    assert bucket({"tags": ["kjxlkj", "confirmation"]}) == "kjxlkj_mutation_confirmation"
    assert bucket_rates(cases)["direct_answer"]["pass_rate"] == 0.5


def test_supervised_labels_mask_non_assistant_tokens():
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": '{"kind":"final","content":"hi"}'},
        ]
    }
    ids, labels = supervised_token_ids(DummyTokenizer(), row)
    assert len(ids) == len(labels)
    assert any(label == -100 for label in labels)
    assert any(label != -100 for label in labels)


def test_raw_user_prompt_becomes_default_task():
    messages = normalize_messages([{"role": "user", "content": "What is 2+3?"}])
    assert "<task>" in messages[0]["content"]
    assert "<request>What is 2+3?</request>" in messages[0]["content"]
    assert "<constraints>Return one valid JSON action.</constraints>" in messages[0]["content"]


def test_latest_user_event_extracts_tagged_context():
    content = '<events><event kind="user">What is lkjai?</event></events>'
    assert latest_user_event(content) == "What is lkjai?"


def test_agent_context_messages_include_tool_observation():
    content = '<events><event kind="user">Search resources.</event><event kind="observation">release-notes</event></events>'
    messages = agent_context_messages(content)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "release-notes"


def test_agent_context_messages_preserve_empty_observation():
    content = '<events><event kind="user">Search resources.</event><event kind="observation"></event></events>'
    messages = agent_context_messages(content)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == ""


def test_prepare_preferences_writes_pairs(tmp_path):
    paths = Paths(str(tmp_path))
    prepare_fixtures(paths)
    out = prepare_preferences(paths)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines
    assert {"messages", "chosen", "rejected", "source"}.issubset(lines[0])


class DummyTokenizer:
    def encode(self, text):
        return Encoded([ord(char) for char in text])


class Encoded:
    def __init__(self, ids):
        self.ids = ids
