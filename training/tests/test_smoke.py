import json

import pytest

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.corpus import generate_corpus
from lkjai_train.dataset import prepare_fixtures, validate_dataset
from lkjai_train.formatting import prompt_text
from lkjai_train.generation import (
    agent_context_messages,
    first_json_object,
    latest_user_event,
    normalize_action,
)
from lkjai_train.paths import Paths
from lkjai_train.preference import prepare_preferences


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
    assert settings.sequence_len == 768
    assert settings.hidden_size == 640
    assert settings.layers == 12
    assert settings.kv_heads == 2
    assert settings.batch_size == 1
    assert settings.corpus_size == 4000


def test_quick_settings_are_tiny():
    settings = train_settings("quick")
    assert settings.sequence_len == 64
    assert settings.hidden_size == 64
    assert settings.max_steps == 5


def test_fixture_dataset_validates(tmp_path):
    paths = Paths(str(tmp_path))
    fixture = prepare_fixtures(paths)
    assert validate_dataset(fixture) == fixture


def test_prompt_text_appends_assistant_header():
    text = prompt_text([{"role": "user", "content": "hello"}])
    assert text.startswith("<bos>")
    assert text.endswith("<assistant>")
    assert "<conversation>" in text


def test_agent_corpus_default_has_required_mix():
    rows = generate_corpus(4000)
    assert len(rows) == 4000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert tags.count("tool_trajectory") >= 1000
    assert tags.count("direct_answer") >= 800
    assert "kjxlkj" in tags
    assert "public_instruction" in tags


def test_normalize_action_extracts_first_json_object():
    text = 'noise {"kind":"final","content":"ok"} trailing'
    assert json.loads(normalize_action(text))["content"] == "ok"
    assert first_json_object(text) == '{"kind":"final","content":"ok"}'


def test_latest_user_event_extracts_agent_context():
    content = "run_id=1\nrecent_events:\nuser: What is 2+2?\nobservation: ignored"
    assert latest_user_event(content) == "What is 2+2?"


def test_latest_user_event_extracts_tagged_context():
    content = '<events><event kind="user">What is lkjai?</event></events>'
    assert latest_user_event(content) == "What is lkjai?"


def test_agent_context_messages_include_tool_observation():
    content = "recent_events:\nuser: List files.\nobservation: README.md"
    messages = agent_context_messages(content)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "README.md"


def test_prepare_preferences_writes_pairs(tmp_path):
    paths = Paths(str(tmp_path))
    out = prepare_preferences(paths)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines
    assert {"messages", "chosen", "rejected", "source"}.issubset(lines[0])
