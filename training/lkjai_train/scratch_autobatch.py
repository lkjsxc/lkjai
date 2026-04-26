import json
import math

import torch

from .checkpoint_state import restore_rng_state, rng_state
from .scratch_optim import autocast_context


def maybe_auto_batch(model, settings, device: torch.device, config) -> None:
    if getattr(settings, "batch_policy", "oom_fallback") == "fixed" or not settings.auto_batch or device.type != "cuda":
        return
    chosen = probe_largest_batch(model, settings, device, config)
    settings.batch_size = chosen
    settings.gradient_accumulation = max(1, math.ceil(settings.target_effective_batch_tokens / max(1, chosen * settings.sequence_len)))
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    print(
        json.dumps(
            {
                "event": "auto_batch",
                "batch_policy": settings.batch_policy,
                "batch_size": settings.batch_size,
                "gradient_accumulation": settings.gradient_accumulation,
                "target_effective_batch_tokens": settings.target_effective_batch_tokens,
                "max_cuda_memory_allocated": memory,
            }
        ),
        flush=True,
    )


def probe_largest_batch(model, settings, device: torch.device, config) -> int:
    saved_rng = rng_state()
    was_training = model.training
    low, high, best = 1, max(1, settings.auto_batch_max), 1
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        while low <= high:
            batch_size = (low + high) // 2
            if probe_batch_fits(model, settings, device, config, batch_size):
                best = batch_size
                low = batch_size + 1
            else:
                high = batch_size - 1
        return best
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
        restore_rng_state(saved_rng)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def probe_batch_fits(model, settings, device: torch.device, config, batch_size: int) -> bool:
    try:
        model.zero_grad(set_to_none=True)
        generator = torch.Generator(device=device)
        generator.manual_seed(settings.seed + batch_size)
        shape = (batch_size, settings.sequence_len)
        input_ids = torch.randint(5, config.vocab_size, shape, device=device, dtype=torch.long, generator=generator)
        labels = torch.randint(5, config.vocab_size, shape, device=device, dtype=torch.long, generator=generator)
        with autocast_context(device, settings.amp):
            _, loss, _ = model(input_ids, labels)
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False
