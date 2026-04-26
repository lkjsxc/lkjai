import json

import torch

from .checkpoint_state import restore_rng_state, rng_state
from .scratch_optim import autocast_context


def compile_model(model, settings, device):
    mode = resolved_compile_mode(settings, device)
    if mode == "off":
        return model
    compiled = torch.compile(model, mode=mode)
    print(json.dumps({"event": "torch_compile", "mode": mode, "static_shapes": settings.static_shapes}), flush=True)
    return compiled


def resolved_compile_mode(settings, device) -> str:
    choice = settings.compile
    if choice == "off" or device.type != "cuda":
        return "off"
    if choice == "auto":
        return "reduce-overhead"
    if choice in {"default", "reduce-overhead"}:
        return choice
    raise ValueError("TRAIN_COMPILE must be off, auto, default, or reduce-overhead")


def warmup_compiled_model(model, settings, device, config) -> None:
    if resolved_compile_mode(settings, device) == "off" or settings.compile_warmup_microsteps <= 0:
        return
    saved_rng = rng_state()
    try:
        for step in range(settings.compile_warmup_microsteps):
            generator = torch.Generator(device=device)
            generator.manual_seed(settings.seed + 1000 + step)
            shape = (settings.batch_size, settings.sequence_len)
            ids = torch.randint(5, config.vocab_size, shape, device=device, dtype=torch.long, generator=generator)
            labels = torch.randint(5, config.vocab_size, shape, device=device, dtype=torch.long, generator=generator)
            with autocast_context(device, settings.amp):
                _, loss, _ = model(ids, labels)
            loss.backward()
            model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(json.dumps({"event": "torch_compile_warmup", "microsteps": settings.compile_warmup_microsteps}), flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng_state(saved_rng)
