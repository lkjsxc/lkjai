import json
import math
import time
from pathlib import Path

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
    profile = StepProfiler(paths.runs / "perf-steps.jsonl", settings.profile_steps, settings.benchmark_warmup_microsteps)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while counters["optimizer_steps"] < settings.max_optimizer_steps and not stop:
        iterator = iter(loader)
        while True:
            wait_start = time.perf_counter()
            try:
                input_ids, labels = next(iterator)
            except StopIteration:
                break
            wait_seconds = time.perf_counter() - wait_start
            loss, event = train_batch(model, input_ids, labels, optimizer, scheduler, scaler, settings, device, counters)
            if loss is None:
                continue
            event["loader_wait_seconds"] = wait_seconds
            train_losses.append(loss)
            if counters["microsteps"] % settings.gradient_accumulation == 0:
                opt_start = time.perf_counter()
                optimizer_step(model, optimizer, scheduler, scaler)
                if device.type == "cuda" and settings.profile_steps:
                    torch.cuda.synchronize()
                event["optimizer_seconds"] = time.perf_counter() - opt_start
                counters["optimizer_steps"] += 1
                log_train_event(counters, loss, start_time, settings, prior_elapsed)
                best_metric = maybe_validate(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, validation_history, best_metric, prior_elapsed, start_time)
            profile.write(counters, settings, event)
            stop = should_stop(counters, settings)
            if stop:
                break
    return finish_loop(model, val_loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, validation_history, best_metric, train_losses, prior_elapsed, start_time)


def train_batch(model, input_ids, labels, optimizer, scheduler, scaler, settings, device, counters):
    event = {"microstep_seconds": 0.0, "h2d_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0}
    sync_for_profile = bool(settings.profile_steps)
    step_start = time.perf_counter()
    h2d_start = time.perf_counter()
    input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    if device.type == "cuda" and sync_for_profile:
        torch.cuda.synchronize()
    event["h2d_seconds"] = time.perf_counter() - h2d_start
    loss_tokens = int((labels != -100).sum().item())
    if loss_tokens == 0:
        return None, event
    forward_start = time.perf_counter()
    with autocast_context(device, settings.amp):
        _, loss, _ = model(input_ids, labels)
    if device.type == "cuda" and sync_for_profile:
        torch.cuda.synchronize()
    event["forward_seconds"] = time.perf_counter() - forward_start
    scaled_loss = loss / settings.gradient_accumulation
    backward_start = time.perf_counter()
    scaler.scale(scaled_loss).backward() if scaler is not None else scaled_loss.backward()
    if device.type == "cuda" and sync_for_profile:
        torch.cuda.synchronize()
    event["backward_seconds"] = time.perf_counter() - backward_start
    counters["microsteps"] += 1
    counters["input_tokens"] += int(input_ids.numel())
    counters["loss_tokens"] += loss_tokens
    event["microstep_seconds"] = time.perf_counter() - step_start
    event["loss"] = float(loss.detach().cpu())
    event["loss_tokens"] = loss_tokens
    event["input_tokens"] = int(input_ids.numel())
    return event["loss"], event


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


class StepProfiler:
    def __init__(self, path: Path, max_steps: int, warmup_microsteps: int):
        self.path = path
        self.max_steps = max_steps
        self.warmup_microsteps = warmup_microsteps
        self.written = 0
        if max_steps:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def write(self, counters: dict, settings, event: dict) -> None:
        if not self.max_steps:
            return
        if counters["microsteps"] <= self.warmup_microsteps:
            return
        if self.written >= self.max_steps:
            return
        record = {
            "microsteps": counters["microsteps"],
            "optimizer_steps": counters["optimizer_steps"],
            "data_mode": settings.data_mode,
            "dataloader_impl": settings.dataloader_impl,
            "batch_size": settings.batch_size,
            "sequence_len": settings.sequence_len,
            "amp": settings.amp,
            "torch_compile": settings.torch_compile,
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.written += 1
