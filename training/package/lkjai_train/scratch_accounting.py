import torch

from .scratch_eval import should_log, should_validate


def new_accounting(device: torch.device) -> dict:
    return {
        "loss_sum": torch.zeros((), device=device),
        "last_loss": torch.zeros((), device=device),
        "loss_count": 0,
        "loss_tokens": torch.zeros((), device=device, dtype=torch.long),
    }


def record_loss(accounting: dict, loss: torch.Tensor, loss_tokens: torch.Tensor) -> None:
    accounting["loss_sum"] = accounting["loss_sum"] + loss
    accounting["last_loss"] = loss
    accounting["loss_count"] += 1
    accounting["loss_tokens"] = accounting["loss_tokens"] + loss_tokens


def flush_accounting(counters: dict, accounting: dict, losses: dict) -> None:
    if accounting["loss_count"]:
        losses["sum"] += float(accounting["loss_sum"].detach().cpu())
        losses["last"] = float(accounting["last_loss"].detach().cpu())
        losses["count"] += accounting["loss_count"]
        accounting["loss_sum"].zero_()
        accounting["loss_count"] = 0
    pending_loss_tokens = int(accounting["loss_tokens"].detach().cpu())
    if pending_loss_tokens:
        counters["loss_tokens"] += pending_loss_tokens
    accounting["loss_tokens"].zero_()


def boundary_needs_accounting(step: int, settings) -> bool:
    return (
        should_log(step, settings)
        or should_validate(step, settings)
        or due(step, getattr(settings, "save_latest_every_optimizer_steps", 0))
        or due(step, getattr(settings, "intermediate_save_every_optimizer_steps", 0))
    )


def due(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0
