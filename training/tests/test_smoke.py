from lkjai_train.cli import dispatch
from lkjai_train.paths import Paths


class Args:
    command = "smoke"


def test_smoke_pipeline(tmp_path):
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert (result / "config.json").exists()
    assert (result / "model.safetensors").exists()
    assert (result / "size.json").exists()


def test_train_command_can_run_tiny_pipeline(tmp_path, monkeypatch):
    class TrainArgs:
        command = "train"

    monkeypatch.setenv("TRAIN_TINY", "1")
    monkeypatch.setenv("TRAIN_TOKEN_BUDGET", "200")
    monkeypatch.setenv("TRAIN_VOCAB_SIZE", "259")
    monkeypatch.setenv("TRAIN_STEPS", "1")
    result = dispatch(TrainArgs(), Paths(str(tmp_path)))
    assert (result / "config.json").exists()
    assert (result / "tokenizer.json").exists()
