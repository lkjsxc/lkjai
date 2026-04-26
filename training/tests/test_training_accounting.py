from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_best_checkpoint_only_saves_on_validation_improvement(monkeypatch, tmp_path):
    import lkjai_train.scratch_eval as scratch_eval

    saved = []

    def fake_evaluate_loss(model, loader, device, settings):
        return {"loss": next(losses), "loss_tokens": 8}

    def fake_save_checkpoint(path, *args, **kwargs):
        saved.append((path, kwargs.get("source_type")))

    losses = iter([3.0, 4.0, 2.0])
    monkeypatch.setattr(scratch_eval, "evaluate_loss", fake_evaluate_loss)
    monkeypatch.setattr(scratch_eval, "save_checkpoint_atomic", fake_save_checkpoint)
    monkeypatch.setattr(scratch_eval, "write_checkpoint_manifest", lambda paths, settings: None)
    paths = SimpleNamespace(checkpoint_best=tmp_path / "best", checkpoint_latest=tmp_path / "latest")
    settings = SimpleNamespace(objective="causal_lm_full")
    history = []
    counters = {"optimizer_steps": 1}

    scratch_eval.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, float("inf"))
    scratch_eval.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, 3.0)
    scratch_eval.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, 3.0)

    assert saved == [
        (paths.checkpoint_best, "best"),
        (paths.checkpoint_latest, "latest"),
        (paths.checkpoint_best, "best"),
        (paths.checkpoint_latest, "latest"),
    ]
    assert [item["loss"] for item in history] == [3.0, 4.0, 2.0]
