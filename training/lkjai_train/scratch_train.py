import json

import torch

from .objectives import normalize_objective
from .scratch_autobatch import maybe_auto_batch, probe_batch_fits, probe_largest_batch
from .scratch_compile import compile_model, warmup_compiled_model
from .scratch_eval import save_training
from .scratch_loaders import cache_paths, loader, val_source
from .scratch_loop import run_loop
from .scratch_model import ModelConfig, ScratchLM
from .scratch_optim import create_optimizer, create_scaler, create_scheduler, maybe_resume
from .tokenizer import load_tokenizer, train_tokenizer


def train_scratch(paths, settings) -> dict:
    paths.ensure()
    validate_settings(settings)
    configure_torch(settings)
    if not paths.tokenizer_json.exists():
        train_tokenizer(paths, settings)
    tokenizer = load_tokenizer(paths.tokenizer_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, settings.seed)
    config = model_config(settings, tokenizer.get_vocab_size())
    model = ScratchLM(config).to(device)
    model.configure_runtime(settings)
    maybe_auto_batch(model, settings, device, config)
    optimizer = create_optimizer(model, settings, device)
    scheduler = create_scheduler(optimizer, settings)
    scheduler._lkjai_clip_grad_norm = settings.clip_grad_norm
    scaler = create_scaler(device, settings.amp)
    pad_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>") or 0
    train_cache, val_cache = cache_paths(paths, tokenizer, settings)
    train_loader = loader(train_cache, settings, pad_id, True, device, config.vocab_size, "train")
    val_loader = loader(val_cache, settings, pad_id, False, device, config.vocab_size, "val")
    state = maybe_resume(paths, settings, model, optimizer, scheduler, scaler, device)
    model = compile_model(model, settings, device)
    warmup_compiled_model(model, settings, device, config)
    metrics = run_loop(model, optimizer, scheduler, scaler, train_loader, val_loader, settings, device, config, paths, state)
    return save_training(paths, settings, config, model, train_cache, val_cache, metrics)


def validate_settings(settings) -> None:
    settings.objective = normalize_objective(settings.objective)
    choices = {
        "TRAIN_EXPORT_CHECKPOINT": (settings.export_checkpoint, {"best", "final"}),
        "TRAIN_AMP": (settings.amp, {"auto", "bf16", "fp16", "off"}),
        "TRAIN_DATA_MODE": (settings.data_mode, {"real", "synthetic_cpu", "synthetic_gpu"}),
        "TRAIN_DATALOADER_IMPL": (settings.dataloader_impl, {"legacy", "mapped"}),
        "TRAIN_LR_SCHEDULE": (settings.lr_schedule, {"cosine", "constant", "linear_warmup_cosine"}),
        "TRAIN_CHECKPOINT_RESUME_SOURCE": (settings.checkpoint_resume_source, {"latest", "final", "best"}),
        "TRAIN_BATCH_POLICY": (settings.batch_policy, {"fixed", "oom_fallback", "sweep"}),
        "TRAIN_ACTIVATION_CHECKPOINT": (settings.activation_checkpoint, {"off", "all", "every_n"}),
        "TRAIN_ATTENTION_BACKEND": (settings.attention_backend, {"auto", "sdpa", "flash2"}),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}")
    validate_non_negative(settings)


def validate_non_negative(settings) -> None:
    minimums = {
        "TRAIN_NUM_WORKERS": settings.num_workers,
        "TRAIN_CLIP_GRAD_NORM": settings.clip_grad_norm,
        "TRAIN_KEEP_LAST_CHECKPOINTS": settings.keep_last_checkpoints,
        "TRAIN_LOG_EVERY_OPTIMIZER_STEPS": settings.log_every_optimizer_steps,
    }
    for name, value in minimums.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    if settings.auto_batch_max < 1:
        raise ValueError("TRAIN_AUTO_BATCH_MAX must be >= 1")
    if settings.activation_checkpoint_every_n < 1:
        raise ValueError("TRAIN_ACTIVATION_CHECKPOINT_EVERY_N must be >= 1")


def configure_torch(settings) -> None:
    torch.backends.cuda.matmul.allow_tf32 = settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = settings.allow_tf32
    if settings.matmul_precision:
        torch.set_float32_matmul_precision(settings.matmul_precision)
    print(json.dumps({"event": "training_config", "model_preset": settings.model_preset, "batch_size": settings.batch_size, "compile": settings.compile}), flush=True)


def seed_torch(device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def model_config(settings, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size,
        settings.sequence_len,
        settings.layers,
        settings.hidden_size,
        settings.heads,
        settings.kv_heads,
        settings.ffn_size,
        settings.dropout,
    )
