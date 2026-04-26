from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


def train_settings(schedule="linear_warmup_cosine"):
    return SimpleNamespace(
        learning_rate=0.01,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_schedule=schedule,
        warmup_steps=2,
        max_optimizer_steps=6,
        lr_min_factor=0.1,
        objective="causal_lm_full",
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_scheduler_supports_constant_schedule():
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_optim import create_optimizer, create_scheduler

    model = ScratchLM(ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0))
    settings = train_settings("constant")
    optimizer = create_optimizer(model, settings, torch.device("cpu"))
    scheduler = create_scheduler(optimizer, settings)

    for _ in range(3):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == settings.learning_rate


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_scheduler_state_resumes_next_lr(tmp_path):
    from lkjai_train.checkpointing import load_checkpoint, save_checkpoint_atomic
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_optim import create_optimizer, create_scheduler

    config = ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)
    settings = train_settings()
    model = ScratchLM(config)
    optimizer = create_optimizer(model, settings, torch.device("cpu"))
    scheduler = create_scheduler(optimizer, settings)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    save_checkpoint_atomic(tmp_path / "ckpt", config, model, optimizer, scheduler, None, {"microsteps": 3, "optimizer_steps": 3, "input_tokens": 24, "loss_tokens": 24}, settings, 1.0, [], source_type="intermediate")

    uninterrupted = optimizer
    uninterrupted_scheduler = scheduler
    uninterrupted.step()
    uninterrupted_scheduler.step()
    expected_lr = uninterrupted.param_groups[0]["lr"]

    restored = ScratchLM(config)
    restored_optimizer = create_optimizer(restored, settings, torch.device("cpu"))
    restored_scheduler = create_scheduler(restored_optimizer, settings)
    load_checkpoint(tmp_path / "ckpt", restored, restored_optimizer, restored_scheduler, None, "cpu")
    restored_optimizer.step()
    restored_scheduler.step()

    assert restored_optimizer.param_groups[0]["lr"] == expected_lr


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_optimizer_excludes_rmsnorm_weights_from_decay():
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_optim import create_optimizer

    model = ScratchLM(ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0))
    optimizer = create_optimizer(model, train_settings(), torch.device("cpu"))
    groups = {group["name"]: group for group in optimizer.param_groups}
    no_decay_ids = {id(param) for param in groups["no_decay"]["params"]}

    assert id(model.norm.weight) in no_decay_ids
    assert groups["decay"]["weight_decay"] == 0.1
    assert groups["no_decay"]["weight_decay"] == 0.0
