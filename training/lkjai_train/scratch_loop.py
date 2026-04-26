import time

import torch

from .checkpointing import prune_old_checkpoints, save_checkpoint_atomic, write_checkpoint_manifest
from .scratch_accounting import boundary_needs_accounting, due, flush_accounting, new_accounting, record_loss
from .scratch_eval import (
    fresh_counters,
    log_train_event,
    optimizer_step,
    should_log,
    should_stop,
    should_validate,
    validate_and_maybe_save,
)
from .scratch_optim import autocast_context
from .scratch_profile import StepProfiler


def run_loop(model, optimizer, scheduler, scaler, loader, val_loader, settings, device: torch.device, config, paths, state: dict) -> dict:
    counters = state.get("counters", fresh_counters())
    prior_elapsed = float(counters.get("elapsed_seconds", 0.0))
    best_metric = float(state.get("best_metric", float("inf")))
    history = list(state.get("validation_history", []))
    losses = {"last": 0.0, "sum": 0.0, "count": 0}
    accounting = new_accounting(device)
    start_time, stop = time.perf_counter(), False
    profile_steps = settings.profile_steps or (100 if getattr(settings, "dataloader_benchmark", False) else 0)
    profile = StepProfiler(paths.runs / "perf-steps.jsonl", profile_steps, settings.benchmark_warmup_microsteps)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while counters["optimizer_steps"] < settings.max_optimizer_steps and not stop:
        iterator = iter(loader)
        while True:
            input_ids, labels, wait_seconds = next_batch(iterator)
            if input_ids is None:
                break
            loss, loss_tokens, event = train_batch(model, input_ids, labels, scaler, settings, device, counters, profile.enabled)
            event["loader_wait_seconds"] = wait_seconds
            record_loss(accounting, loss, loss_tokens)
            if counters["microsteps"] % settings.gradient_accumulation == 0:
                best_metric = optimizer_boundary(model, optimizer, scheduler, scaler, val_loader, settings, device, config, paths, counters, history, best_metric, prior_elapsed, start_time, accounting, losses, event)
            profile.write(counters, settings, event)
            stop = should_stop(counters, settings)
            if stop:
                break
    return finish_loop(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best_metric, losses, accounting, prior_elapsed, start_time)


def next_batch(iterator):
    wait_start = time.perf_counter()
    try:
        input_ids, labels = next(iterator)
    except StopIteration:
        return None, None, time.perf_counter() - wait_start
    return input_ids, labels, time.perf_counter() - wait_start


def train_batch(model, input_ids, labels, scaler, settings, device, counters, profile_enabled: bool):
    event = {"microstep_seconds": 0.0, "h2d_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0}
    step_start = time.perf_counter()
    h2d_start = time.perf_counter()
    input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    sync_profile(device, profile_enabled)
    event["h2d_seconds"] = time.perf_counter() - h2d_start
    forward_start = time.perf_counter()
    with autocast_context(device, settings.amp):
        _, loss, _ = model(input_ids, labels)
    sync_profile(device, profile_enabled)
    event["forward_seconds"] = time.perf_counter() - forward_start
    scaled_loss = loss / settings.gradient_accumulation
    backward_start = time.perf_counter()
    scaler.scale(scaled_loss).backward() if scaler is not None else scaled_loss.backward()
    sync_profile(device, profile_enabled)
    event["backward_seconds"] = time.perf_counter() - backward_start
    counters["microsteps"] += 1
    counters["input_tokens"] += input_ids.numel()
    event["microstep_seconds"] = time.perf_counter() - step_start
    loss_tokens = (labels != -100).sum()
    if profile_enabled:
        event.update({"loss": float(loss.detach().cpu()), "loss_tokens": int(loss_tokens.detach().cpu()), "input_tokens": input_ids.numel()})
    return loss.detach(), loss_tokens.detach(), event


def optimizer_boundary(model, optimizer, scheduler, scaler, val_loader, settings, device, config, paths, counters, history, best, prior, start, accounting, losses, event):
    opt_start = time.perf_counter()
    optimizer_step(model, optimizer, scheduler, scaler)
    sync_profile(device, bool(settings.profile_steps))
    event["optimizer_seconds"] = time.perf_counter() - opt_start
    counters["optimizer_steps"] += 1
    if boundary_needs_accounting(counters["optimizer_steps"], settings):
        flush_accounting(counters, accounting, losses)
    if should_log(counters["optimizer_steps"], settings):
        log_train_event(counters, losses["last"], start, settings, optimizer, device, prior)
    best = maybe_validate(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best, prior, start, accounting, losses)
    maybe_save_snapshots(model, optimizer, scheduler, scaler, settings, config, paths, counters, history, best, prior, start, accounting, losses)
    return best


def maybe_validate(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best, prior, start, accounting, losses):
    if should_validate(counters["optimizer_steps"], settings):
        flush_accounting(counters, accounting, losses)
        counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
        metric = validate_and_maybe_save(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best)
        return min(best, metric["loss"])
    return best


def maybe_save_snapshots(model, optimizer, scheduler, scaler, settings, config, paths, counters, history, best, prior, start, accounting, losses) -> None:
    step = counters["optimizer_steps"]
    save_latest = due(step, getattr(settings, "save_latest_every_optimizer_steps", 0))
    save_intermediate = due(step, getattr(settings, "intermediate_save_every_optimizer_steps", 0))
    if not save_latest and not save_intermediate:
        return
    flush_accounting(counters, accounting, losses)
    counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
    if save_latest:
        save_checkpoint_atomic(paths.checkpoint_latest, config, model, optimizer, scheduler, scaler, counters, settings, best, history, source_type="latest")
    if save_intermediate:
        path = paths.checkpoint_steps / f"step-{step:06d}"
        save_checkpoint_atomic(path, config, model, optimizer, scheduler, scaler, counters, settings, best, history, source_type="intermediate")
        prune_old_checkpoints(paths, settings.keep_last_checkpoints)
    write_checkpoint_manifest(paths, settings)


def finish_loop(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best, losses, accounting, prior, start):
    flush_accounting(counters, accounting, losses)
    counters["elapsed_seconds"] = prior + (time.perf_counter() - start)
    final = validate_and_maybe_save(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best)
    best, elapsed = min(best, final["loss"]), max(1e-9, counters["elapsed_seconds"])
    save_checkpoint_atomic(paths.checkpoint_final, config, model, optimizer, scheduler, scaler, counters, settings, best, history, source_type="final", validation_loss=final["loss"])
    save_checkpoint_atomic(paths.checkpoint_latest, config, model, optimizer, scheduler, scaler, counters, settings, best, history, source_type="latest", validation_loss=final["loss"])
    write_checkpoint_manifest(paths, settings)
    return summary_metrics(counters, settings, losses, final, best, elapsed, history)


def summary_metrics(counters, settings, losses, final, best, elapsed, history) -> dict:
    return {
        "train_loss": losses["last"],
        "mean_train_loss": losses["sum"] / max(1, losses["count"]),
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


def sync_profile(device, enabled: bool) -> None:
    if device.type == "cuda" and enabled:
        torch.cuda.synchronize()
