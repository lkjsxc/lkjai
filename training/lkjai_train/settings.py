from dataclasses import dataclass

from .model_presets import model_shape
from .settings_env import env_bool, env_float, env_int, env_str


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
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    lr_schedule: str
    warmup_steps: int
    lr_min_factor: float
    gradient_checkpointing: bool
    activation_checkpoint: str
    activation_checkpoint_every_n: int
    checkpoint_preserve_rng: bool
    batch_size: int
    gradient_accumulation: int
    batch_policy: str
    auto_batch: bool
    auto_batch_max: int
    target_effective_batch_tokens: int
    max_optimizer_steps: int
    max_microsteps: int
    max_steps: int
    validate_every_optimizer_steps: int
    save_every_optimizer_steps: int
    intermediate_save_every_optimizer_steps: int
    keep_last_checkpoints: int
    save_latest_every_optimizer_steps: int
    checkpoint_resume_source: str
    log_every_optimizer_steps: int
    validation_batches: int
    resume: str
    amp: str
    compile: str
    compile_warmup_microsteps: int
    static_shapes: bool
    attention_backend: str
    export_checkpoint: str
    fixed_eval_threshold: float
    behavioral_threshold: float
    enforce_competency: bool
    corpus_size: int
    corpus_tokens: int
    corpus_dir: str
    curriculum: str
    seed: int
    data_mode: str
    dataloader_impl: str
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
    allow_tf32: bool
    matmul_precision: str
    clip_grad_norm: float
    profile_steps: int
    benchmark_warmup_microsteps: int
    dataloader_benchmark: bool


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return quick_settings()
    allowed = {"agent", "custom", "scratch-20m", "scratch-30m-debug", "scratch-60m", "scratch-93m-max"}
    if preset not in allowed:
        raise ValueError(f"unknown TRAIN_PRESET={preset}")
    model_preset = env_str("TRAIN_MODEL_PRESET", "scratch-20m" if preset in {"agent", "custom"} else preset)
    return settings(preset, model_preset, *model_shape(model_preset))


def quick_settings() -> TrainSettings:
    max_steps = optimizer_steps_default(5)
    batch = env_int("TRAIN_BATCH_SIZE", 1)
    accumulation = env_int("TRAIN_GRADIENT_ACCUMULATION", 1)
    sequence_len = env_int("TRAIN_SEQUENCE_LEN", 64)
    return make_settings("quick", "quick", *model_shape("quick"), max_steps=max_steps, batch=batch, accumulation=accumulation, seq_override=sequence_len)


def settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows):
    return make_settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows)


def make_settings(preset, model_preset, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows=20, max_steps=None, batch=None, accumulation=None, seq_override=None):
    max_steps = optimizer_steps_default(steps if max_steps is None else max_steps)
    sequence_len = env_int("TRAIN_SEQUENCE_LEN", seq_override or seq)
    batch_size = env_int("TRAIN_BATCH_SIZE", batch if batch is not None else default_batch(preset))
    gradient_accumulation = env_int("TRAIN_GRADIENT_ACCUMULATION", accumulation if accumulation is not None else default_accumulation(preset))
    checkpoint = env_str("TRAIN_ACTIVATION_CHECKPOINT", default_checkpoint_mode(preset))
    return TrainSettings(
        preset=preset,
        model_name=env_str("MODEL_NAME", default_model_name(model_preset)),
        model_preset=model_preset,
        objective=env_str("TRAIN_OBJECTIVE", "causal_lm_full"),
        vocab_size=env_int("TRAIN_VOCAB_SIZE", vocab),
        sequence_len=sequence_len,
        layers=env_int("TRAIN_LAYERS", layers),
        hidden_size=env_int("TRAIN_HIDDEN_SIZE", hidden),
        heads=env_int("TRAIN_HEADS", heads),
        kv_heads=env_int("TRAIN_KV_HEADS", kv),
        ffn_size=env_int("TRAIN_FFN_SIZE", ffn),
        dropout=env_float("TRAIN_DROPOUT", 0.0),
        learning_rate=env_float("TRAIN_LEARNING_RATE", 3e-4),
        weight_decay=env_float("TRAIN_WEIGHT_DECAY", 0.01),
        beta1=env_float("TRAIN_BETA1", 0.9),
        beta2=env_float("TRAIN_BETA2", 0.999),
        eps=env_float("TRAIN_EPS", 1e-8),
        lr_schedule=env_str("TRAIN_LR_SCHEDULE", "linear_warmup_cosine"),
        warmup_steps=env_int("TRAIN_WARMUP_STEPS", min(100, max(1, max_steps // 10))),
        lr_min_factor=env_float("TRAIN_LR_MIN_FACTOR", 0.1),
        gradient_checkpointing=checkpoint != "off",
        activation_checkpoint=checkpoint,
        activation_checkpoint_every_n=env_int("TRAIN_ACTIVATION_CHECKPOINT_EVERY_N", 2),
        checkpoint_preserve_rng=env_bool("TRAIN_CHECKPOINT_PRESERVE_RNG", False),
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        batch_policy=env_str("TRAIN_BATCH_POLICY", "fixed"),
        auto_batch=env_bool("TRAIN_AUTO_BATCH", False),
        auto_batch_max=env_int("TRAIN_AUTO_BATCH_MAX", 16),
        target_effective_batch_tokens=env_int("TRAIN_TARGET_EFFECTIVE_BATCH_TOKENS", batch_size * sequence_len * gradient_accumulation),
        max_optimizer_steps=max_steps,
        max_microsteps=env_int("TRAIN_MAX_MICROSTEPS", 0),
        max_steps=max_steps,
        validate_every_optimizer_steps=env_int("TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS", 1 if preset == "quick" else 250),
        save_every_optimizer_steps=env_int("TRAIN_SAVE_EVERY_OPTIMIZER_STEPS", 0),
        intermediate_save_every_optimizer_steps=env_int("TRAIN_INTERMEDIATE_SAVE_EVERY_OPTIMIZER_STEPS", 0 if preset == "quick" else 1000),
        keep_last_checkpoints=env_int("TRAIN_KEEP_LAST_CHECKPOINTS", 3),
        save_latest_every_optimizer_steps=env_int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS", 1 if preset == "quick" else 250),
        checkpoint_resume_source=env_str("TRAIN_CHECKPOINT_RESUME_SOURCE", "latest"),
        log_every_optimizer_steps=env_int("TRAIN_LOG_EVERY_OPTIMIZER_STEPS", 250),
        validation_batches=env_int("TRAIN_VALIDATION_BATCHES", 2 if preset == "quick" else 8),
        resume=env_str("TRAIN_RESUME", "auto"),
        amp=env_str("TRAIN_AMP", "auto"),
        compile=env_str("TRAIN_COMPILE", "off" if preset == "quick" else "auto"),
        compile_warmup_microsteps=env_int("TRAIN_COMPILE_WARMUP_MICROSTEPS", 2),
        static_shapes=env_bool("TRAIN_STATIC_SHAPES", True),
        attention_backend=env_str("TRAIN_ATTENTION_BACKEND", "auto"),
        export_checkpoint=env_str("TRAIN_EXPORT_CHECKPOINT", "best"),
        fixed_eval_threshold=env_float("TRAIN_FIXED_EVAL_THRESHOLD", 0.6 if preset != "quick" else 0.0),
        behavioral_threshold=env_float("TRAIN_BEHAVIORAL_THRESHOLD", 0.35 if preset != "quick" else 0.0),
        enforce_competency=env_bool("TRAIN_ENFORCE_COMPETENCY", False),
        corpus_size=env_int("TRAIN_CORPUS_SIZE", rows),
        corpus_tokens=env_int("TRAIN_CORPUS_TOKENS", 500_000_000),
        corpus_dir=env_str("TRAIN_CORPUS_DIR", "/app/data/kimi-corpus"),
        curriculum=env_str("TRAIN_CURRICULUM", "configs/curriculum/agent_20m.toml"),
        seed=env_int("TRAIN_SEED", 42),
        data_mode=env_str("TRAIN_DATA_MODE", "real"),
        dataloader_impl=env_str("TRAIN_DATALOADER_IMPL", "legacy" if preset == "quick" else "mapped"),
        num_workers=env_int("TRAIN_NUM_WORKERS", 0),
        pin_memory=env_bool("TRAIN_PIN_MEMORY", True),
        prefetch_factor=env_int("TRAIN_PREFETCH_FACTOR", 2),
        persistent_workers=env_bool("TRAIN_PERSISTENT_WORKERS", False),
        allow_tf32=env_bool("TRAIN_ALLOW_TF32", True),
        matmul_precision=env_str("TRAIN_MATMUL_PRECISION", "high"),
        clip_grad_norm=env_float("TRAIN_CLIP_GRAD_NORM", 1.0),
        profile_steps=env_int("TRAIN_PROFILE_STEPS", 0),
        benchmark_warmup_microsteps=env_int("TRAIN_BENCHMARK_WARMUP_MICROSTEPS", 0),
        dataloader_benchmark=env_bool("TRAIN_DATALOADER_BENCHMARK", False),
    )


def default_model_name(model_preset: str) -> str:
    return "lkjai-scratch-20m" if model_preset == "scratch-20m" else f"lkjai-{model_preset}"


def default_batch(preset: str) -> int:
    return 1 if preset == "quick" else 2


def default_accumulation(preset: str) -> int:
    return 1 if preset == "quick" else 4


def default_checkpoint_mode(preset: str) -> str:
    return "off" if preset in {"quick", "agent", "custom", "scratch-20m"} else "all"


def optimizer_steps_default(default: int) -> int:
    return env_int("TRAIN_MAX_OPTIMIZER_STEPS", env_int("TRAIN_MAX_STEPS", default))
