from lkjai_train.cli import dispatch
from lkjai_train.paths import Paths


class Args:
    command = "smoke"


def test_smoke_pipeline(tmp_path):
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert (result / "config.json").exists()
    assert (result / "model.safetensors").exists()
    assert (result / "size.json").exists()
