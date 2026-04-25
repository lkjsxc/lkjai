from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_best_checkpoint_only_saves_on_validation_improvement(monkeypatch, tmp_path):
    import lkjai_train.scratch_train as scratch_train

    saved = []

    def fake_evaluate_loss(model, loader, device, settings):
        return {"loss": next(losses), "loss_tokens": 8}

    def fake_save_checkpoint(path, *args):
        saved.append(path)

    losses = iter([3.0, 4.0, 2.0])
    monkeypatch.setattr(scratch_train, "evaluate_loss", fake_evaluate_loss)
    monkeypatch.setattr(scratch_train, "save_checkpoint", fake_save_checkpoint)
    paths = SimpleNamespace(checkpoint_best=tmp_path / "best")
    settings = SimpleNamespace(objective="causal_lm_full")
    history = []
    counters = {"optimizer_steps": 1}

    scratch_train.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, float("inf"))
    scratch_train.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, 3.0)
    scratch_train.validate_and_maybe_save(None, None, "cpu", settings, None, paths, None, None, None, counters, history, 3.0)

    assert saved == [paths.checkpoint_best, paths.checkpoint_best]
    assert [item["loss"] for item in history] == [3.0, 4.0, 2.0]

