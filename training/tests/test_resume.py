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

