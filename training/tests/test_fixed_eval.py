import json

import pytest

from lkjai_train.cli import dispatch
from lkjai_train.evals import evaluate_fixed_suite
from lkjai_train.paths import Paths


torch = None
tokenizers = None
try:
    import torch
    import tokenizers
except ImportError:
    pass


class SmokeArgs:
    command = "smoke"


@pytest.mark.slow
@pytest.mark.skipif(torch is None or tokenizers is None, reason="training deps not installed")
def test_fixed_eval_report_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_MAX_STEPS", "1")
    paths = Paths(str(tmp_path))
    dispatch(SmokeArgs(), paths)
    report_path = evaluate_fixed_suite(paths, 0.8)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["threshold"] == 0.8
    assert 0.0 <= report["pass_rate"] <= 1.0
    assert report["total"] == len(report["cases"])
    assert all("id" in item and "passed" in item for item in report["cases"])
