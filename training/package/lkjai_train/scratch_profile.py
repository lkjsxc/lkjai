import json
from pathlib import Path


class StepProfiler:
    def __init__(self, path: Path, max_steps: int, warmup_microsteps: int):
        self.path = path
        self.max_steps = max_steps
        self.warmup_microsteps = warmup_microsteps
        self.written = 0
        self.enabled = bool(max_steps)
        if max_steps:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def write(self, counters: dict, settings, event: dict) -> None:
        if not self.max_steps or counters["microsteps"] <= self.warmup_microsteps:
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
            "compile": settings.compile,
            "activation_checkpoint": settings.activation_checkpoint,
            "attention_backend": settings.attention_backend,
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.written += 1
