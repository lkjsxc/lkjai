import json
import math
import time

import torch

from .checkpointing import save_checkpoint
from .scratch_eval import fresh_counters, log_train_event, optimizer_step, should_save, should_stop, should_validate, validate_and_maybe_save
from .scratch_optim import autocast_context


def run_loop(model, optimizer, scheduler, scaler, loader, val_loader, settings, device: torch.device, config, paths, state: dict) -> dict:
    counters = state.get("counters", fresh_counters())
    prior_elapsed = float(counters.get("elapsed_seconds", 0.0))
    best_metric = float(state.get("best_metric", float("inf")))
    validation_history = list(state.get("validation_history", []))
    train_losses, start_time, stop = [], time.perf_counter(), False
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while counters["optimizer_steps"] < settings.max_optimizer_steps and not stop:
        for input_ids, labels in loader:
            loss = train_batch(model, input_ids, labels, optimizer, scheduler, scaler, settings, device, counters)
            if loss is None:
                continue
            train_losses.append(loss)
            if counters["microsteps"] % settings.gradient_accumulation == 0:
                optimizer_step(model, optimizer, scheduler, scaler)
                counters["optimizer_steps"] += 1
                log_train_event(counters, loss, start_time, settings, prior_elapsed)
                best_metric = maybe_validate(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, validation_history, best_metric, prior_elapsed, start_time)
            stop = should_stop(counters, settings)
            if stop:
                break
    return finish_loop(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, validation_history, best_metric, train_losses, prior_elapsed, start_time)


def train_batch(model, input_ids, labels, optimizer, scheduler, scaler, settings, device, counters):
    input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    loss_tokens = int((labels != -100).sum().item())
    if loss_tokens == 0:
        return None
    with autocast_context(device, settings.amp):
        _, loss, _ = model(input_ids, labels)
    scaled_loss = loss / settings.gradient_accumulation
    scaler.scale(scaled_loss).backward() if scaler is not None else scaled_loss.backward()
    counters["microsteps"] += 1
    counters["input_tokens"] += int(input_ids.numel())
    counters["loss_tokens"] += loss_tokens
    return float(loss.detach().cpu())


def maybe_validate(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best, prior, start):
    if should_validate(counters["optimizer_steps"], settings):
        counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
        metric = validate_and_maybe_save(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best)
        return min(best, metric["loss"])
    if should_save(counters["optimizer_steps"], settings):
        counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
        save_checkpoint(paths.checkpoint_final, config, model, optimizer, scheduler, scaler, counters, settings, best, history)
    return best


def finish_loop(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best, losses, prior, start):
    counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
    final = validate_and_maybe_save(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best)
    best, elapsed = min(best, final["loss"]), max(1e-9, counters["elapsed_seconds"])
    save_checkpoint(paths.checkpoint_final, config, model, optimizer, scheduler, scaler, counters, settings, best, history)
    return {
        "train_loss": losses[-1] if losses else 0.0,
        "mean_train_loss": sum(losses) / max(1, len(losses)),
        "validation_loss": final["loss"],
        "validation_perplexity": final.get("perplexity"),
        "best_validation_loss": best,
        "microsteps": counters["microsteps"],
        "optimizer_steps": counters["optimizer_steps"],
        "gradient_accumulation": settings.gradient_accumulation,
        "input_tokens_seen": counters["input_tokens"],
        "loss_tokens_seen": counters["loss_tokens"],
        "elapsed_seconds": elapsed,
        "tokens_per_second": counters["input_tokens"] / elapsed,
        "loss_tokens_per_second": counters["loss_tokens"] / elapsed,
        "effective_batch_input_tokens_per_optimizer_step": settings.batch_size * settings.sequence_len * settings.gradient_accumulation,
        "effective_batch_loss_tokens_per_optimizer_step": counters["loss_tokens"] / max(1, counters["optimizer_steps"]),
        "validation_history": history,
    }
