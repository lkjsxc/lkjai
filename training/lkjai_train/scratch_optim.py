import math
from contextlib import nullcontext

import torch

from .checkpointing import checkpoint_exists, load_checkpoint


def create_optimizer(model, settings, device: torch.device):
    kwargs = {"lr": settings.learning_rate}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def create_scaler(device: torch.device, amp: str):
    if device.type == "cuda" and amp == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda":
        return nullcontext()
    if amp == "bf16" or (amp == "auto" and torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def lr_lambda(max_optimizer_steps: int):
    warmup = min(100, max(1, max_optimizer_steps // 10))

    def schedule(step: int):
        if step < warmup:
            return max(0.1, (step + 1) / warmup)
        progress = (step - warmup) / max(1, max_optimizer_steps - warmup)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return schedule


def maybe_resume(paths, settings, model, optimizer, scheduler, scaler, device) -> dict:
    mode = settings.resume
    if mode not in {"auto", "never", "required"}:
        raise ValueError("TRAIN_RESUME must be auto, never, or required")
    exists = checkpoint_exists(paths.checkpoint_final)
    if mode == "required" and not exists:
        raise RuntimeError(f"resume required but no checkpoint exists at {paths.checkpoint_final}")
    if mode == "never" or not exists:
        return {}
    return load_checkpoint(paths.checkpoint_final, model, optimizer, scheduler, scaler, device)
