import torch
from torch.utils.data import DataLoader

from .objectives import normalize_objective
from .packed_data import PackedDataset, build_or_load_packed_cache
from .packed_datasets import MappedPackedDataset, SyntheticPackedDataset
from .checkpointing import save_checkpoint
from .scratch_eval import evaluate_loss, save_training
from .scratch_loop import run_loop
from .scratch_model import ModelConfig, ScratchLM
from .scratch_optim import create_optimizer, create_scaler, lr_lambda, maybe_resume
from .tokenizer import load_tokenizer, train_source, train_tokenizer


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
    model.enable_gradient_checkpointing(settings.gradient_checkpointing)
    optimizer = create_optimizer(model, settings, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(settings.max_optimizer_steps))
    scheduler._lkjai_clip_grad_norm = settings.clip_grad_norm
    scaler = create_scaler(device, settings.amp)
    pad_id = tokenizer.token_to_id("<eos>") or tokenizer.token_to_id("<pad>") or 0
    train_cache, val_cache = cache_paths(paths, tokenizer, settings)
    train_loader = loader(train_cache, settings, pad_id, True, device, config.vocab_size, "train")
    val_loader = loader(val_cache, settings, pad_id, False, device, config.vocab_size, "val")
    state = maybe_resume(paths, settings, model, optimizer, scheduler, scaler, device)
    if settings.torch_compile:
        model = torch.compile(model, mode=settings.torch_compile_mode)
    metrics = run_loop(model, optimizer, scheduler, scaler, train_loader, val_loader, settings, device, config, paths, state)
    return save_training(paths, settings, config, model, train_cache, val_cache, metrics)


def validate_settings(settings) -> None:
    settings.objective = normalize_objective(settings.objective)
    if settings.export_checkpoint not in {"best", "final"}:
        raise ValueError("TRAIN_EXPORT_CHECKPOINT must be best or final")
    if settings.amp not in {"auto", "bf16", "fp16", "off"}:
        raise ValueError("TRAIN_AMP must be auto, bf16, fp16, or off")
    if settings.data_mode not in {"real", "synthetic_cpu", "synthetic_gpu"}:
        raise ValueError("TRAIN_DATA_MODE must be real, synthetic_cpu, or synthetic_gpu")
    if settings.dataloader_impl not in {"legacy", "mapped"}:
        raise ValueError("TRAIN_DATALOADER_IMPL must be legacy or mapped")
    if settings.num_workers < 0:
        raise ValueError("TRAIN_NUM_WORKERS must be >= 0")
    if settings.clip_grad_norm < 0:
        raise ValueError("TRAIN_CLIP_GRAD_NORM must be >= 0")


def configure_torch(settings) -> None:
    torch.backends.cuda.matmul.allow_tf32 = settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = settings.allow_tf32
    if settings.matmul_precision:
        torch.set_float32_matmul_precision(settings.matmul_precision)


def seed_torch(device, seed: int) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def cache_paths(paths, tokenizer, settings):
    if settings.data_mode != "real":
        return None, None
    train_src, val_src = train_source(paths), val_source(paths)
    train_cache = build_or_load_packed_cache(paths, tokenizer, train_src, "train", settings)
    val_cache = build_or_load_packed_cache(paths, tokenizer, val_src, "val", settings)
    return train_cache, val_cache


def loader(cache, settings, pad_id: int, shuffle: bool, device, vocab_size: int, split: str):
    if settings.data_mode == "synthetic_gpu" and device.type == "cuda":
        return SyntheticGpuLoader(settings, vocab_size, split, device)
    if settings.data_mode == "synthetic_cpu":
        dataset = SyntheticPackedDataset(synthetic_windows(settings, split), settings.sequence_len, vocab_size, settings.seed)
        return DataLoader(dataset, batch_size=settings.batch_size, shuffle=shuffle, pin_memory=device.type == "cuda")
    dataset_cls = MappedPackedDataset if settings.dataloader_impl == "mapped" else PackedDataset
    kwargs = {
        "batch_size": settings.batch_size,
        "shuffle": shuffle,
        "pin_memory": device.type == "cuda" and settings.pin_memory,
        "num_workers": settings.num_workers,
    }
    if settings.num_workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
        kwargs["persistent_workers"] = settings.persistent_workers
    return DataLoader(
        dataset_cls(cache, settings.sequence_len, pad_id),
        **kwargs,
    )


def synthetic_windows(settings, split: str) -> int:
    steps = max(settings.max_microsteps, settings.max_optimizer_steps * settings.gradient_accumulation)
    multiplier = 2 if split == "train" else 1
    return max(settings.validation_batches + 1, steps * settings.batch_size * multiplier)


class SyntheticGpuLoader:
    def __init__(self, settings, vocab_size: int, split: str, device: torch.device):
        self.settings = settings
        self.vocab_size = vocab_size
        self.split = split
        self.device = device
        self.windows = synthetic_windows(settings, split)

    def __len__(self):
        return self.windows

    def __iter__(self):
        generator = torch.Generator(device=self.device)
        offset = 0 if self.split == "train" else 10_000_000
        generator.manual_seed(self.settings.seed + offset)
        shape = (self.settings.batch_size, self.settings.sequence_len + 1)
        for _ in range(self.windows):
            ids = torch.randint(5, self.vocab_size, shape, device=self.device, dtype=torch.long, generator=generator)
            yield ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


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
