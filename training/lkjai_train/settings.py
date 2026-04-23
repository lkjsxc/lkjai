import os
from dataclasses import dataclass


@dataclass
class TrainSettings:
    preset: str
    model_name: str
    model_preset: str
    vocab_size: int
    sequence_len: int
    layers: int
    hidden_size: int
    heads: int
    kv_heads: int
    ffn_size: int
    learning_rate: float
    gradient_checkpointing: bool
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    fixed_eval_threshold: float
    enforce_competency: bool
    corpus_size: int
    seed: int


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return quick_settings()
    if preset in {"agent", "custom"}:
        return settings(
            preset,
            env_str("TRAIN_MODEL_PRESET", "scratch-60m"),
            8192,
            768,
            12,
            640,
            8,
            2,
            1792,
            3000,
            4000,
        )
    raise ValueError(f"unknown TRAIN_PRESET={preset}")


def quick_settings() -> TrainSettings:
    return TrainSettings(
        "quick",
        env_str("MODEL_NAME", "lkjai-scratch-60m"),
        "tiny-scratch",
        512,
        64,
        2,
        64,
        4,
        2,
        128,
        3e-4,
        False,
        1,
        1,
        5,
        env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
        env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        20,
        env_int("TRAIN_SEED", 42),
    )


def settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows):
    return TrainSettings(
        preset=preset,
        model_name=env_str("MODEL_NAME", "lkjai-scratch-60m"),
        model_preset=model_preset,
        vocab_size=env_int("TRAIN_VOCAB_SIZE", vocab),
        sequence_len=env_int("TRAIN_SEQUENCE_LEN", seq),
        layers=env_int("TRAIN_LAYERS", layers),
        hidden_size=env_int("TRAIN_HIDDEN_SIZE", hidden),
        heads=env_int("TRAIN_HEADS", heads),
        kv_heads=env_int("TRAIN_KV_HEADS", kv),
        ffn_size=env_int("TRAIN_FFN_SIZE", ffn),
        learning_rate=env_float("TRAIN_LEARNING_RATE", 3e-4),
        gradient_checkpointing=env_bool("TRAIN_GRADIENT_CHECKPOINTING", True),
        batch_size=env_int("TRAIN_BATCH_SIZE", 1),
        gradient_accumulation=env_int("TRAIN_GRADIENT_ACCUMULATION", 8),
        max_steps=env_int("TRAIN_MAX_STEPS", steps),
        fixed_eval_threshold=env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.8),
        enforce_competency=env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        corpus_size=env_int("TRAIN_CORPUS_SIZE", rows),
        seed=env_int("TRAIN_SEED", 42),
    )


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}
