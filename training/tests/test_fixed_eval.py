import json

from lkjai_train.cli import dispatch
from lkjai_train.evals import evaluate_fixed_suite
from lkjai_train.paths import Paths


class SmokeArgs:
    command = "smoke"


def test_fixed_eval_report_schema(tmp_path):
    paths = Paths(str(tmp_path))
    dispatch(SmokeArgs(), paths)
    report_path = evaluate_fixed_suite(paths, 0.8)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["threshold"] == 0.8
    assert 0.0 <= report["pass_rate"] <= 1.0
    assert report["total"] == len(report["cases"])
    assert all("id" in item and "passed" in item for item in report["cases"])
