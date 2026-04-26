from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


def settings():
    return SimpleNamespace(objective="causal_lm_full", keep_last_checkpoints=2)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_atomic_checkpoint_round_trip_and_metadata(tmp_path):
    from lkjai_train.checkpointing import load_checkpoint, save_checkpoint_atomic
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    model = ScratchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    counters = {"microsteps": 2, "optimizer_steps": 2, "input_tokens": 16, "loss_tokens": 16}

    save_checkpoint_atomic(tmp_path / "step-000002", config, model, optimizer, scheduler, None, counters, settings(), 1.5, [{"loss": 1.5}], source_type="intermediate")

    restored = ScratchLM(config)
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_sched = torch.optim.lr_scheduler.LambdaLR(restored_optim, lambda step: 1.0)
    state = load_checkpoint(tmp_path / "step-000002", restored, restored_optim, restored_sched, None, "cpu")

    assert state["counters"] == counters
    assert (tmp_path / "step-000002" / "metadata.json").exists()


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_latest_ignores_incomplete_temp_checkpoint(tmp_path):
    from lkjai_train.checkpointing import latest_complete_checkpoint, save_checkpoint_atomic
    from lkjai_train.paths import Paths
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    paths = Paths(str(tmp_path))
    paths.ensure()
    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    model = ScratchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    counters = {"microsteps": 1, "optimizer_steps": 1, "input_tokens": 8, "loss_tokens": 8}

    (paths.checkpoints / ".latest.tmp-broken").mkdir()
    save_checkpoint_atomic(paths.checkpoint_latest, config, model, optimizer, scheduler, None, counters, settings(), 2.0, [], source_type="latest")

    assert latest_complete_checkpoint(paths, "latest") == paths.checkpoint_latest


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_intermediate_checkpoint_retention_prunes_oldest(tmp_path):
    from lkjai_train.checkpointing import prune_old_checkpoints, retained_intermediate_checkpoints, save_checkpoint_atomic
    from lkjai_train.paths import Paths
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    paths = Paths(str(tmp_path))
    paths.ensure()
    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    model = ScratchLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    for step in [1, 2, 3]:
        counters = {"microsteps": step, "optimizer_steps": step, "input_tokens": 8 * step, "loss_tokens": 8 * step}
        save_checkpoint_atomic(paths.checkpoint_steps / f"step-{step:06d}", config, model, optimizer, scheduler, None, counters, settings(), 2.0, [], source_type="intermediate")

    retained = prune_old_checkpoints(paths, 2)

    assert [path.name for path in retained] == ["step-000002", "step-000003"]
    assert [path.name for path in retained_intermediate_checkpoints(paths)] == ["step-000002", "step-000003"]
