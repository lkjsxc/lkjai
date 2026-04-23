import json

import pytest

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.dataset import prepare_fixtures, validate_dataset
from lkjai_train.formatting import prompt_text
from lkjai_train.generation import first_json_object, normalize_action
from lkjai_train.paths import Paths


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
    assert settings.model_preset == "scratch-40m"
    assert settings.sequence_len == 1024
    assert settings.hidden_size == 512
    assert settings.kv_heads == 2
    assert settings.batch_size == 1
    assert settings.corpus_size == 200


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
    assert text.endswith("<|assistant|>")


def test_normalize_action_extracts_first_json_object():
    text = 'noise {"kind":"final","content":"ok"} trailing'
    assert json.loads(normalize_action(text))["content"] == "ok"
    assert first_json_object(text) == '{"kind":"final","content":"ok"}'
