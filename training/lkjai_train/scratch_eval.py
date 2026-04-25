import json
import math
import time

import torch

from .checkpointing import save_checkpoint, unwrap_model
from .scratch_model import parameter_count
from .scratch_optim import autocast_context


def fresh_counters() -> dict:
    return {"microsteps": 0, "optimizer_steps": 0, "input_tokens": 0, "loss_tokens": 0, "elapsed_seconds": 0.0}


def optimizer_step(model, optimizer, scheduler, scaler) -> None:
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer) if scaler is not None else optimizer.step()
    if scaler is not None:
        scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def should_validate(optimizer_steps: int, settings) -> bool:
    interval = settings.validate_every_optimizer_steps
    return interval > 0 and optimizer_steps > 0 and optimizer_steps % interval == 0


def should_save(optimizer_steps: int, settings) -> bool:
    interval = settings.save_every_optimizer_steps
    return interval > 0 and optimizer_steps > 0 and optimizer_steps % interval == 0


def should_stop(counters: dict, settings) -> bool:
    if counters["optimizer_steps"] >= settings.max_optimizer_steps:
        return True
    return bool(settings.max_microsteps and counters["microsteps"] >= settings.max_microsteps)


def validate_and_maybe_save(model, loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best: float) -> dict:
    metric = evaluate_loss(model, loader, device, settings)
    metric["optimizer_steps"] = counters["optimizer_steps"]
    history.append(metric)
    if metric["loss"] < best:
        save_checkpoint(paths.checkpoint_best, config, model, optimizer, scheduler, scaler, counters, settings, metric["loss"], history)
    return metric


@torch.inference_mode()
def evaluate_loss(model, loader, device: torch.device, settings) -> dict:
    model.eval()
    losses, loss_tokens = [], 0
    for index, (input_ids, labels) in enumerate(loader):
        if index >= settings.validation_batches:
            break
        input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_loss_tokens = int((labels != -100).sum().item())
        if batch_loss_tokens == 0:
            continue
        with autocast_context(device, settings.amp):
            _, loss, _ = model(input_ids, labels)
        losses.append(float(loss.detach().cpu()))
        loss_tokens += batch_loss_tokens
    model.train()
    metric = {"loss": sum(losses) / max(1, len(losses)), "loss_tokens": loss_tokens}
    if settings.objective == "causal_lm_full":
        metric["perplexity"] = math.exp(min(20.0, metric["loss"]))
    return metric


def log_train_event(counters: dict, loss: float, start_time: float, settings, prior_elapsed: float = 0.0) -> None:
    step = counters["optimizer_steps"]
    if step == 1 or step % 250 == 0:
        elapsed = max(1e-9, prior_elapsed + (time.perf_counter() - start_time))
        print(json.dumps({"event": "train_step", "optimizer_step": step, "microsteps": counters["microsteps"], "loss": loss, "objective": settings.objective, "input_tokens_seen": counters["input_tokens"], "loss_tokens_seen": counters["loss_tokens"], "tokens_per_second": counters["input_tokens"] / elapsed, "loss_tokens_per_second": counters["loss_tokens"] / elapsed}), flush=True)


def save_training(paths, settings, config, model, train_cache, val_cache, metrics: dict) -> dict:
    params = parameter_count(unwrap_model(model))
    corpus_train_tokens = read_train_tokens(paths)
    tpp = corpus_train_tokens / max(1, params)
    summary = {"backend": "tiny-pytorch-scratch-v3", "checkpoint_dir": str(paths.checkpoint_best if settings.export_checkpoint == "best" else paths.checkpoint_final), "final_checkpoint_dir": str(paths.checkpoint_final), "best_checkpoint_dir": str(paths.checkpoint_best), "objective": settings.objective, "max_steps_semantics": "optimizer_steps", "parameter_count": params, "corpus_train_tokens": corpus_train_tokens, "tokens_per_parameter": round(tpp, 6), "chinchilla_gap": round(max(0.0, 1.0 - tpp / 20.0), 6), "train_cache": str(train_cache), "val_cache": str(val_cache), "settings": settings.__dict__, "metrics": metrics}
    paths.training_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_train_tokens(paths) -> int:
    if paths.tokenizer_manifest.exists():
        return int(json.loads(paths.tokenizer_manifest.read_text(encoding="utf-8")).get("train_tokens", 0))
    return 0
