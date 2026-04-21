from lkjai_train.cli import dispatch, train_settings
from lkjai_train.paths import Paths


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
    settings = train_settings("longrun")
    assert settings.max_duration_secs == 21_600
    assert settings.enforce_competency is True
