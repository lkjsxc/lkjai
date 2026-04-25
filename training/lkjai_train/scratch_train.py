import torch
from torch.utils.data import DataLoader

from .objectives import normalize_objective
from .packed_data import PackedDataset, build_or_load_packed_cache
from .checkpointing import save_checkpoint
from .scratch_eval import evaluate_loss, save_training
from .scratch_loop import run_loop
from .scratch_model import ModelConfig, ScratchLM
from .scratch_optim import create_optimizer, create_scaler, lr_lambda, maybe_resume
from .tokenizer import load_tokenizer, train_source, train_tokenizer


def train_scratch(paths, settings) -> dict:
    paths.ensure()
    validate_settings(settings)
    if not paths.tokenizer_json.exists():
        train_tokenizer(paths, settings)
    tokenizer = load_tokenizer(paths.tokenizer_json)
    train_src, val_src = train_source(paths), val_source(paths)
    train_cache = build_or_load_packed_cache(paths, tokenizer, train_src, "train", settings)
    val_cache = build_or_load_packed_cache(paths, tokenizer, val_src, "val", settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, settings.seed)
    config = model_config(settings, tokenizer.get_vocab_size())
    model = ScratchLM(config).to(device)
    model.enable_gradient_checkpointing(settings.gradient_checkpointing)
    optimizer = create_optimizer(model, settings, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(settings.max_optimizer_steps))
    scaler = create_scaler(device, settings.amp)
    pad_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>") or 0
    train_loader = loader(train_cache, settings, pad_id, shuffle=True, device=device)
    val_loader = loader(val_cache, settings, pad_id, shuffle=False, device=device)
    state = maybe_resume(paths, settings, model, optimizer, scheduler, scaler, device)
    if settings.torch_compile:
        model = torch.compile(model)
    metrics = run_loop(model, optimizer, scheduler, scaler, train_loader, val_loader, settings, device, config, paths, state)
    return save_training(paths, settings, config, model, train_cache, val_cache, metrics)


def validate_settings(settings) -> None:
    settings.objective = normalize_objective(settings.objective)
    if settings.export_checkpoint not in {"best", "final"}:
        raise ValueError("TRAIN_EXPORT_CHECKPOINT must be best or final")
    if settings.amp not in {"auto", "bf16", "fp16"}:
        raise ValueError("TRAIN_AMP must be auto, bf16, or fp16")


def seed_torch(device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def loader(cache, settings, pad_id: int, shuffle: bool, device):
    return DataLoader(
        PackedDataset(cache, settings.sequence_len, pad_id),
        batch_size=settings.batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
    )


def val_source(paths):
    if paths.committed_val.exists() and any(paths.committed_val.rglob("*.jsonl")):
        return paths.committed_val
    return paths.val_dataset if paths.val_dataset.exists() else paths.fixtures


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


def validate_and_maybe_save(model, loader, device, settings, config, paths, optimizer, scheduler, scaler, counters, history, best_metric: float) -> dict:
    metric = evaluate_loss(model, loader, device, settings)
    metric["optimizer_steps"] = counters["optimizer_steps"]
    history.append(metric)
    if metric["loss"] < best_metric:
        save_checkpoint(paths.checkpoint_best, config, model, optimizer, scheduler, scaler, counters, settings, metric["loss"], history)
    return metric
