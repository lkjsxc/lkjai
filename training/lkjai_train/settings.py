import os
from dataclasses import dataclass


@dataclass
class TrainSettings:
    preset: str
    model_name: str
    model_preset: str
    objective: str
    vocab_size: int
    sequence_len: int
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    ffn_size: int
    dropout: float
    learning_rate: float
    gradient_checkpointing: bool
    batch_size: int
    gradient_accumulation: int
    max_optimizer_steps: int
    max_microsteps: int
    max_steps: int
    validate_every_optimizer_steps: int
    save_every_optimizer_steps: int
    validation_batches: int
    resume: str
    amp: str
    torch_compile: bool
    export_checkpoint: str
    fixed_eval_threshold: float
    behavioral_threshold: float
    enforce_competency: bool
    corpus_size: int
    corpus_tokens: int
    corpus_dir: str
    seed: int


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return quick_settings()
    if preset in {"agent", "custom", "scratch-30m-debug", "scratch-60m", "scratch-93m-max"}:
        model_preset = env_str("TRAIN_MODEL_PRESET", "scratch-60m" if preset in {"agent", "custom"} else preset)
        return settings(preset, model_preset, *model_shape(model_preset))
    raise ValueError(f"unknown TRAIN_PRESET={preset}")


def quick_settings() -> TrainSettings:
    max_steps = optimizer_steps_default(5)
    return TrainSettings(
        "quick",
        env_str("MODEL_NAME", "lkjai-scratch-60m"),
        "quick",
        env_str("TRAIN_OBJECTIVE", "causal_lm_full"),
        env_int("TRAIN_VOCAB_SIZE", 512),
        env_int("TRAIN_SEQUENCE_LEN", 64),
        env_int("TRAIN_LAYERS", 2),
        env_int("TRAIN_HIDDEN_SIZE", 64),
        env_int("TRAIN_HEADS", 4),
        env_int("TRAIN_KV_HEADS", 2),
        env_int("TRAIN_FFN_SIZE", 128),
        env_float("TRAIN_DROPOUT", 0.0),
        env_float("TRAIN_LEARNING_RATE", 3e-4),
        env_bool("TRAIN_GRADIENT_CHECKPOINTING", False),
        env_int("TRAIN_BATCH_SIZE", 1),
        env_int("TRAIN_GRADIENT_ACCUMULATION", 1),
        max_steps,
        env_int("TRAIN_MAX_MICROSTEPS", 0),
        max_steps,
        env_int("TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS", 1),
        env_int("TRAIN_SAVE_EVERY_OPTIMIZER_STEPS", 0),
        env_int("TRAIN_VALIDATION_BATCHES", 2),
        env_str("TRAIN_RESUME", "auto"),
        env_str("TRAIN_AMP", "auto"),
        env_bool("TRAIN_TORCH_COMPILE", False),
        env_str("TRAIN_EXPORT_CHECKPOINT", "best"),
        0.0,
        0.0,
        False,
        20,
        0,
        "",
        env_int("TRAIN_SEED", 42),
    )


def settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows):
    max_steps = optimizer_steps_default(steps)
    return TrainSettings(
        preset=preset,
        model_name=env_str("MODEL_NAME", "lkjai-scratch-60m"),
        model_preset=model_preset,
        objective=env_str("TRAIN_OBJECTIVE", "causal_lm_full"),
        vocab_size=env_int("TRAIN_VOCAB_SIZE", vocab),
        sequence_len=env_int("TRAIN_SEQUENCE_LEN", seq),
        layers=env_int("TRAIN_LAYERS", layers),
        hidden_size=env_int("TRAIN_HIDDEN_SIZE", hidden),
        heads=env_int("TRAIN_HEADS", heads),
        kv_heads=env_int("TRAIN_KV_HEADS", kv),
        ffn_size=env_int("TRAIN_FFN_SIZE", ffn),
        dropout=env_float("TRAIN_DROPOUT", 0.0),
        learning_rate=env_float("TRAIN_LEARNING_RATE", 3e-4),
        gradient_checkpointing=env_bool("TRAIN_GRADIENT_CHECKPOINTING", True),
        batch_size=env_int("TRAIN_BATCH_SIZE", 1),
        gradient_accumulation=env_int("TRAIN_GRADIENT_ACCUMULATION", 8),
        max_optimizer_steps=max_steps,
        max_microsteps=env_int("TRAIN_MAX_MICROSTEPS", 0),
        max_steps=max_steps,
        validate_every_optimizer_steps=env_int("TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS", 250),
        save_every_optimizer_steps=env_int("TRAIN_SAVE_EVERY_OPTIMIZER_STEPS", 1000),
        validation_batches=env_int("TRAIN_VALIDATION_BATCHES", 8),
        resume=env_str("TRAIN_RESUME", "auto"),
        amp=env_str("TRAIN_AMP", "auto"),
        torch_compile=env_bool("TRAIN_TORCH_COMPILE", False),
        export_checkpoint=env_str("TRAIN_EXPORT_CHECKPOINT", "best"),
        fixed_eval_threshold=env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.6),
        behavioral_threshold=env_float("TRAIN_BEHAVIORAL_THRESHOLD", 0.35),
        enforce_competency=env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        corpus_size=env_int("TRAIN_CORPUS_SIZE", rows),
        corpus_tokens=env_int("TRAIN_CORPUS_TOKENS", 500_000_000),
        corpus_dir=env_str("TRAIN_CORPUS_DIR", "/app/data/kimi-corpus"),
        seed=env_int("TRAIN_SEED", 42),
    )


def model_shape(model_preset: str) -> tuple[int, int, int, int, int, int, int, int, int]:
    shapes = {
        "quick": (512, 64, 2, 64, 4, 2, 128, 5, 20),
        "tiny-scratch": (512, 64, 2, 64, 4, 2, 128, 5, 20),
        "scratch-30m-debug": (8192, 1024, 8, 512, 8, 2, 1664, 20000, 120000),
        "scratch-60m": (8192, 1024, 12, 640, 8, 2, 1792, 120000, 120000),
        "scratch-93m-max": (8192, 1024, 15, 768, 12, 2, 1920, 120000, 120000),
    }
    if model_preset not in shapes:
        raise ValueError(f"unknown TRAIN_MODEL_PRESET={model_preset}")
    return shapes[model_preset]


def optimizer_steps_default(default: int) -> int:
    return env_int("TRAIN_MAX_OPTIMIZER_STEPS", env_int("TRAIN_MAX_STEPS", default))


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}
