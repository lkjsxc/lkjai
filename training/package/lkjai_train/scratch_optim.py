import json
from contextlib import nullcontext

import torch

from .checkpoint_files import checkpoint_candidates, checkpoint_exists
from .checkpointing import load_checkpoint
from .scratch_model import RMSNorm


def create_optimizer(model, settings, device: torch.device):
    kwargs = {
        "lr": settings.learning_rate,
        "betas": (settings.beta1, settings.beta2),
        "eps": settings.eps,
        "weight_decay": settings.weight_decay,
    }
    param_groups = parameter_groups(model)
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(param_groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(param_groups, **kwargs)


def parameter_groups(model):
    decay, no_decay, seen = [], [], set()
    norm_types = (torch.nn.LayerNorm, RMSNorm)
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name == "bias" or param.ndim == 1 or isinstance(module, norm_types):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
    ]


def create_scheduler(optimizer, settings):
    schedule = settings.lr_schedule
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, settings.max_optimizer_steps),
            eta_min=settings.learning_rate * settings.lr_min_factor,
        )
    if schedule == "linear_warmup_cosine":
        warmup_steps = max(1, settings.warmup_steps)
        cosine_steps = max(1, settings.max_optimizer_steps - warmup_steps)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(10, warmup_steps),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=settings.learning_rate * settings.lr_min_factor,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    raise ValueError("TRAIN_LR_SCHEDULE must be cosine, constant, or linear_warmup_cosine")


def create_scaler(device: torch.device, amp: str):
    if device.type == "cuda" and amp == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda":
        return nullcontext()
    if amp == "off":
        return nullcontext()
    if amp == "bf16" or (amp == "auto" and torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def maybe_resume(paths, settings, model, optimizer, scheduler, scaler, device) -> dict:
    mode = settings.resume
    if mode not in {"auto", "never", "required"}:
        raise ValueError("TRAIN_RESUME must be auto, never, or required")
    if mode == "never":
        return {}
    # The dataloader position is not tracked; resume restores optimizer,
    # scheduler, scaler, RNG, counters, and metrics exactly, then starts a fresh
    # loader iterator.
    found_checkpoint = False
    for checkpoint_dir in checkpoint_candidates(paths, settings.checkpoint_resume_source):
        if not checkpoint_exists(checkpoint_dir):
            continue
        found_checkpoint = True
        try:
            state = load_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, device)
        except (RuntimeError, ValueError) as error:
            if mode == "required":
                raise RuntimeError(f"resume checkpoint is incompatible: {checkpoint_dir}") from error
            log_resume_skip(checkpoint_dir, error)
            continue
        state["checkpoint_dir"] = str(checkpoint_dir)
        return state
    if not found_checkpoint and mode == "required":
        raise RuntimeError(f"resume required but no complete {settings.checkpoint_resume_source} checkpoint exists")
    return {}


def log_resume_skip(checkpoint_dir, error: Exception) -> None:
    print(
        json.dumps(
            {
                "event": "resume_skip",
                "checkpoint_dir": str(checkpoint_dir),
                "reason": "incompatible_checkpoint",
                "error": str(error).splitlines()[0],
            }
        ),
        flush=True,
    )
