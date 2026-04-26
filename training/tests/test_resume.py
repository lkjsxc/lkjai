from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_checkpoint_restore_counters_optimizer_scheduler_and_best(tmp_path):
    from lkjai_train.checkpointing import load_checkpoint, save_checkpoint
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    model = ScratchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    counters = {"microsteps": 3, "optimizer_steps": 1, "input_tokens": 24, "loss_tokens": 20}
    settings = SimpleNamespace(objective="causal_lm_full")
    save_checkpoint(tmp_path, config, model, optimizer, scheduler, None, counters, settings, 2.5, [{"loss": 2.5}])

    restored = ScratchLM(config)
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_optim, lambda step: 1.0)
    state = load_checkpoint(tmp_path, restored, restored_optim, restored_sched, None, "cpu")

    assert state["counters"] == counters
    assert state["best_metric"] == 2.5
    assert state["validation_history"] == [{"loss": 2.5}]
    assert restored_optim.state_dict()["param_groups"][0]["lr"] == optimizer.state_dict()["param_groups"][0]["lr"]


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_resume_prefers_latest_complete_checkpoint(tmp_path):
    from lkjai_train.checkpointing import save_checkpoint_atomic
    from lkjai_train.paths import Paths
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_optim import maybe_resume

    paths = Paths(str(tmp_path))
    paths.ensure()
    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    model = ScratchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    settings = SimpleNamespace(objective="causal_lm_full", resume="auto", checkpoint_resume_source="latest")

    save_checkpoint_atomic(paths.checkpoint_final, config, model, optimizer, scheduler, None, {"microsteps": 1, "optimizer_steps": 1, "input_tokens": 8, "loss_tokens": 8}, settings, 3.0, [{"loss": 3.0}], source_type="final")
    save_checkpoint_atomic(paths.checkpoint_latest, config, model, optimizer, scheduler, None, {"microsteps": 3, "optimizer_steps": 3, "input_tokens": 24, "loss_tokens": 24}, settings, 2.0, [{"loss": 2.0}], source_type="latest")

    restored = ScratchLM(config)
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_optim, lambda step: 1.0)
    state = maybe_resume(paths, settings, restored, restored_optim, restored_sched, None, "cpu")

    assert state["counters"]["optimizer_steps"] == 3
    assert state["best_metric"] == 2.0


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_auto_resume_skips_incompatible_checkpoint(tmp_path):
    from lkjai_train.checkpointing import save_checkpoint_atomic
    from lkjai_train.paths import Paths
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_optim import maybe_resume

    paths = Paths(str(tmp_path))
    paths.ensure()
    old_config = ModelConfig(32, 8, 2, 16, 4, 2, 32, 0.0)
    old_model = ScratchLM(old_config)
    optimizer = torch.optim.AdamW(old_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    settings = SimpleNamespace(objective="causal_lm_full", resume="auto", checkpoint_resume_source="latest")
    counters = {"microsteps": 1, "optimizer_steps": 1, "input_tokens": 8, "loss_tokens": 8}

    save_checkpoint_atomic(paths.checkpoint_latest, old_config, old_model, optimizer, scheduler, None, counters, settings, 2.0, [], source_type="latest")

    new_config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    new_model = ScratchLM(new_config)
    new_optim = torch.optim.AdamW(new_model.parameters(), lr=0.01)
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_optim, lambda step: 1.0)

    assert maybe_resume(paths, settings, new_model, new_optim, new_sched, None, "cpu") == {}
