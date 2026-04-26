from .model_presets import model_shape
from .settings_config import resolver_for
from .settings_types import TrainSettings


def train_settings(preset: str) -> TrainSettings:
    if preset == "quick":
        return quick_settings()
    allowed = {"agent", "custom", "scratch-20m", "scratch-30m-debug", "scratch-40m", "scratch-60m", "scratch-93m-max"}
    if preset not in allowed:
        raise ValueError(f"unknown TRAIN_PRESET={preset}")
    resolver = resolver_for(preset)
    default_preset = "scratch-40m" if preset in {"agent", "custom"} else preset
    model_preset = resolver.str("TRAIN_MODEL_PRESET", "model_preset", default_preset)
    return settings(preset, model_preset, resolver, *model_shape(model_preset))


def quick_settings() -> TrainSettings:
    resolver = resolver_for("quick")
    max_steps = optimizer_steps_default(5, resolver)
    batch = resolver.int("TRAIN_BATCH_SIZE", "batch_size", 1)
    accumulation = resolver.int("TRAIN_GRADIENT_ACCUMULATION", "gradient_accumulation", 1)
    sequence_len = resolver.int("TRAIN_SEQUENCE_LEN", "sequence_len", 64)
    return make_settings("quick", "quick", resolver, *model_shape("quick"), max_steps=max_steps, batch=batch, accumulation=accumulation, seq_override=sequence_len)


def settings(preset, model_preset, resolver, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows):
    return make_settings(preset, model_preset, resolver, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows)


def make_settings(preset, model_preset, resolver, vocab, seq, layers, hidden, heads, kv, ffn, steps, rows=20, max_steps=None, batch=None, accumulation=None, seq_override=None):
    max_steps = optimizer_steps_default(steps if max_steps is None else max_steps, resolver)
    sequence_len = resolver.int("TRAIN_SEQUENCE_LEN", "sequence_len", seq_override or seq)
    batch_size = resolver.int("TRAIN_BATCH_SIZE", "batch_size", batch if batch is not None else default_batch(preset))
    gradient_accumulation = resolver.int("TRAIN_GRADIENT_ACCUMULATION", "gradient_accumulation", accumulation if accumulation is not None else default_accumulation(preset))
    checkpoint = resolver.str("TRAIN_ACTIVATION_CHECKPOINT", "activation_checkpoint", default_checkpoint_mode(model_preset))
    return TrainSettings(
        preset=preset,
        model_name=resolver.str("MODEL_NAME", "model_name", default_model_name(model_preset)),
        model_preset=model_preset,
        objective=resolver.str("TRAIN_OBJECTIVE", "objective", "causal_lm_full"),
        vocab_size=resolver.int("TRAIN_VOCAB_SIZE", "vocab_size", vocab),
        sequence_len=sequence_len,
        layers=resolver.int("TRAIN_LAYERS", "layers", layers),
        hidden_size=resolver.int("TRAIN_HIDDEN_SIZE", "hidden_size", hidden),
        heads=resolver.int("TRAIN_HEADS", "heads", heads),
        kv_heads=resolver.int("TRAIN_KV_HEADS", "kv_heads", kv),
        ffn_size=resolver.int("TRAIN_FFN_SIZE", "ffn_size", ffn),
        dropout=resolver.float("TRAIN_DROPOUT", "dropout", 0.0),
        learning_rate=resolver.float("TRAIN_LEARNING_RATE", "learning_rate", 3e-4),
        weight_decay=resolver.float("TRAIN_WEIGHT_DECAY", "weight_decay", 0.01),
        beta1=resolver.float("TRAIN_BETA1", "beta1", 0.9),
        beta2=resolver.float("TRAIN_BETA2", "beta2", 0.999),
        eps=resolver.float("TRAIN_EPS", "eps", 1e-8),
        lr_schedule=resolver.str("TRAIN_LR_SCHEDULE", "lr_schedule", "linear_warmup_cosine"),
        warmup_steps=resolver.int("TRAIN_WARMUP_STEPS", "warmup_steps", min(100, max(1, max_steps // 10))),
        lr_min_factor=resolver.float("TRAIN_LR_MIN_FACTOR", "lr_min_factor", 0.1),
        gradient_checkpointing=checkpoint != "off",
        activation_checkpoint=checkpoint,
        activation_checkpoint_every_n=resolver.int("TRAIN_ACTIVATION_CHECKPOINT_EVERY_N", "activation_checkpoint_every_n", 2),
        checkpoint_preserve_rng=resolver.bool("TRAIN_CHECKPOINT_PRESERVE_RNG", "checkpoint_preserve_rng", False),
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        batch_policy=resolver.str("TRAIN_BATCH_POLICY", "batch_policy", default_batch_policy(preset)),
        auto_batch=resolver.bool("TRAIN_AUTO_BATCH", "auto_batch", preset != "quick"),
        auto_batch_max=resolver.int("TRAIN_AUTO_BATCH_MAX", "auto_batch_max", 16),
        target_effective_batch_tokens=resolver.int("TRAIN_TARGET_EFFECTIVE_BATCH_TOKENS", "target_effective_batch_tokens", batch_size * sequence_len * gradient_accumulation),
        max_optimizer_steps=max_steps,
        max_microsteps=resolver.int("TRAIN_MAX_MICROSTEPS", "max_microsteps", 0),
        max_steps=max_steps,
        validate_every_optimizer_steps=resolver.int("TRAIN_VALIDATE_EVERY_OPTIMIZER_STEPS", "validate_every_optimizer_steps", 1 if preset == "quick" else 3000),
        save_every_optimizer_steps=resolver.int("TRAIN_SAVE_EVERY_OPTIMIZER_STEPS", "save_every_optimizer_steps", 0),
        intermediate_save_every_optimizer_steps=resolver.int("TRAIN_INTERMEDIATE_SAVE_EVERY_OPTIMIZER_STEPS", "intermediate_save_every_optimizer_steps", 0 if preset == "quick" else 18000),
        keep_last_checkpoints=resolver.int("TRAIN_KEEP_LAST_CHECKPOINTS", "keep_last_checkpoints", 8),
        save_latest_every_optimizer_steps=resolver.int("TRAIN_SAVE_LATEST_EVERY_OPTIMIZER_STEPS", "save_latest_every_optimizer_steps", 1 if preset == "quick" else 3000),
        checkpoint_resume_source=resolver.str("TRAIN_CHECKPOINT_RESUME_SOURCE", "checkpoint_resume_source", "latest"),
        log_every_optimizer_steps=resolver.int("TRAIN_LOG_EVERY_OPTIMIZER_STEPS", "log_every_optimizer_steps", 3000),
        validation_batches=resolver.int("TRAIN_VALIDATION_BATCHES", "validation_batches", 2 if preset == "quick" else 8),
        resume=resolver.str("TRAIN_RESUME", "resume", "auto"),
        amp=resolver.str("TRAIN_AMP", "amp", "auto"),
        compile=resolver.str("TRAIN_COMPILE", "compile", "off" if preset == "quick" else "auto"),
        compile_warmup_microsteps=resolver.int("TRAIN_COMPILE_WARMUP_MICROSTEPS", "compile_warmup_microsteps", 2),
        static_shapes=resolver.bool("TRAIN_STATIC_SHAPES", "static_shapes", True),
        attention_backend=resolver.str("TRAIN_ATTENTION_BACKEND", "attention_backend", "auto"),
        export_checkpoint=resolver.str("TRAIN_EXPORT_CHECKPOINT", "export_checkpoint", "best"),
        fixed_eval_threshold=resolver.float("TRAIN_FIXED_EVAL_THRESHOLD", "fixed_eval_threshold", 0.6 if preset != "quick" else 0.0),
        behavioral_threshold=resolver.float("TRAIN_BEHAVIORAL_THRESHOLD", "behavioral_threshold", 0.35 if preset != "quick" else 0.0),
        enforce_competency=resolver.bool("TRAIN_ENFORCE_COMPETENCY", "enforce_competency", False),
        corpus_size=resolver.int("TRAIN_CORPUS_SIZE", "corpus_size", rows),
        corpus_tokens=resolver.int("TRAIN_CORPUS_TOKENS", "corpus_tokens", 500_000_000),
        corpus_dir=resolver.str("TRAIN_CORPUS_DIR", "corpus_dir", "/app/data/kimi-corpus"),
        curriculum=resolver.str("TRAIN_CURRICULUM", "curriculum", "configs/curriculum/agent_40m.toml"),
        seed=resolver.int("TRAIN_SEED", "seed", 42),
        data_mode=resolver.str("TRAIN_DATA_MODE", "data_mode", "real"),
        dataloader_impl=resolver.str("TRAIN_DATALOADER_IMPL", "dataloader_impl", "legacy" if preset == "quick" else "mapped"),
        num_workers=resolver.int("TRAIN_NUM_WORKERS", "num_workers", 0),
        pin_memory=resolver.bool("TRAIN_PIN_MEMORY", "pin_memory", True),
        prefetch_factor=resolver.int("TRAIN_PREFETCH_FACTOR", "prefetch_factor", 2),
        persistent_workers=resolver.bool("TRAIN_PERSISTENT_WORKERS", "persistent_workers", False),
        allow_tf32=resolver.bool("TRAIN_ALLOW_TF32", "allow_tf32", True),
        matmul_precision=resolver.str("TRAIN_MATMUL_PRECISION", "matmul_precision", "high"),
        clip_grad_norm=resolver.float("TRAIN_CLIP_GRAD_NORM", "clip_grad_norm", 1.0),
        profile_steps=resolver.int("TRAIN_PROFILE_STEPS", "profile_steps", 0),
        benchmark_warmup_microsteps=resolver.int("TRAIN_BENCHMARK_WARMUP_MICROSTEPS", "benchmark_warmup_microsteps", 0),
        dataloader_benchmark=resolver.bool("TRAIN_DATALOADER_BENCHMARK", "dataloader_benchmark", False),
    )


def default_model_name(model_preset: str) -> str:
    return "lkjai-scratch-20m" if model_preset == "scratch-20m" else f"lkjai-{model_preset}"


def default_batch(preset: str) -> int:
    return 1 if preset == "quick" else 2


def default_accumulation(preset: str) -> int:
    return 1 if preset == "quick" else 4


def default_batch_policy(preset: str) -> str:
    return "fixed" if preset == "quick" else "oom_fallback"


def default_checkpoint_mode(model_preset: str) -> str:
    return "off" if model_preset in {"quick", "scratch-20m"} else "every_n"


def optimizer_steps_default(default: int, resolver) -> int:
    max_steps = resolver.int("TRAIN_MAX_STEPS", "max_steps", default)
    return resolver.int("TRAIN_MAX_OPTIMIZER_STEPS", "max_optimizer_steps", max_steps)
