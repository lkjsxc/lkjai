from types import SimpleNamespace

import pytest


torch = None
try:
    import torch
except ImportError:
    pass


def settings(**overrides):
    base = {
        "auto_batch": True,
        "auto_batch_max": 8,
        "target_effective_batch_tokens": 64,
        "sequence_len": 8,
        "gradient_accumulation": 1,
        "batch_size": 1,
        "amp": "off",
        "seed": 42,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_auto_batch_is_noop_on_cpu():
    from lkjai_train.scratch_train import maybe_auto_batch

    config = SimpleNamespace(vocab_size=32)
    current = settings(batch_size=2, gradient_accumulation=4)
    maybe_auto_batch(None, current, torch.device("cpu"), config)

    assert current.batch_size == 2
    assert current.gradient_accumulation == 4


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_probe_largest_batch_uses_binary_search(monkeypatch):
    import lkjai_train.scratch_autobatch as scratch_autobatch
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    model = ScratchLM(ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0))
    tried = []

    def fake_probe(model, settings, device, config, batch_size):
        tried.append(batch_size)
        return batch_size <= 3

    monkeypatch.setattr(scratch_autobatch, "probe_batch_fits", fake_probe)

    assert scratch_autobatch.probe_largest_batch(model, settings(), torch.device("cpu"), model.config) == 3
    assert tried


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_probe_largest_batch_respects_effective_token_target(monkeypatch):
    import lkjai_train.scratch_autobatch as scratch_autobatch
    from lkjai_train.scratch_model import ModelConfig, ScratchLM

    model = ScratchLM(ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0))
    monkeypatch.setattr(scratch_autobatch, "probe_batch_fits", lambda *args: True)

    current = settings(auto_batch_max=8, target_effective_batch_tokens=32, sequence_len=8)

    assert scratch_autobatch.probe_largest_batch(model, current, torch.device("cpu"), model.config) == 4


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="cuda not available")
def test_cuda_auto_batch_probe_selects_at_least_one():
    from lkjai_train.scratch_model import ModelConfig, ScratchLM
    from lkjai_train.scratch_train import probe_largest_batch

    device = torch.device("cuda")
    model = ScratchLM(ModelConfig(32, 8, 1, 16, 4, 2, 32, 0.0)).to(device)

    assert probe_largest_batch(model, settings(auto_batch_max=1), device, model.config) == 1
