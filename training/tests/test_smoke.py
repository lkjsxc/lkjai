import json

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.packer import pack_tokens
from lkjai_train.paths import Paths
from lkjai_train.tokenizer import train_tokenizer


class Args:
    command = "smoke"


def test_smoke_pipeline(tmp_path):
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert (result / "config.json").exists()
    assert (result / "model.safetensors").exists()
    assert (result / "size.json").exists()


def test_train_command_defaults_to_quick_pipeline(tmp_path, monkeypatch):
    class TrainArgs:
        command = "train"

    monkeypatch.delenv("TRAIN_PRESET", raising=False)
    result = dispatch(TrainArgs(), Paths(str(tmp_path)))
    assert (result / "config.json").exists()
    assert (result / "tokenizer.json").exists()


def test_longrun_settings_defaults(monkeypatch):
    monkeypatch.setenv("TRAIN_PRESET", "longrun")
    monkeypatch.delenv("TRAIN_MAX_DURATION_SECS", raising=False)
    monkeypatch.delenv("TRAIN_ENFORCE_COMPETENCY", raising=False)
    monkeypatch.delenv("TRAIN_TOKENIZER_SAMPLE_CHARS", raising=False)
    settings = train_settings("longrun")
    assert settings.max_duration_secs == 21_600
    assert settings.enforce_competency is True
    assert settings.tokenizer_sample_chars == 5_000_000


def test_pack_tokens_writes_streaming_metadata(tmp_path):
    class CorpusArgs:
        command = "prepare-corpus"
        token_budget = 200
        dataset = "fixture"
        tiny = True

    paths = Paths(str(tmp_path))
    dispatch(CorpusArgs(), paths)
    train_tokenizer(paths, 259, 0)
    out = pack_tokens(paths)
    metadata = json.loads((paths.tokenized / "metadata.json").read_text(encoding="utf-8"))
    assert out.name == "tokens.u16"
    assert metadata["token_file"] == "tokens.u16"
    assert metadata["tokens"] > 0
