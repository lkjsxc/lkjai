import json

import pytest

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.dataset import prepare_fixtures, validate_dataset
from lkjai_train.paths import Paths


torch = None
try:
    import torch
except ImportError:
    pass


class Args:
    command = "smoke"


@pytest.mark.slow
@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_smoke_pipeline(tmp_path):
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert result.name == "fixed-eval.json"
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["pass_rate"] == 1.0
    assert (tmp_path / "adapters" / "manifest.json").exists()
    assert (tmp_path / "adapters" / "final").exists()


def test_smoke_pipeline_without_training_deps(tmp_path):
    """Validate that smoke fails gracefully when training deps are missing."""
    if torch is not None:
        pytest.skip("torch is installed")
    with pytest.raises(RuntimeError):
        dispatch(Args(), Paths(str(tmp_path)))


def test_agent_settings_defaults(monkeypatch):
    monkeypatch.delenv("TRAIN_BASE_MODEL", raising=False)
    monkeypatch.delenv("TRAIN_BATCH_SIZE", raising=False)
    settings = train_settings("agent")
    assert settings.base_model == "Qwen/Qwen3-0.6B"
    assert settings.sequence_len == 2048
    assert settings.lora_rank == 16
    assert settings.load_in_4bit is True
    assert settings.batch_size == 1
    assert settings.corpus_size == 200


def test_fixture_dataset_validates(tmp_path):
    paths = Paths(str(tmp_path))
    fixture = prepare_fixtures(paths)
    assert validate_dataset(fixture) == fixture
