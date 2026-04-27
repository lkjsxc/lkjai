import torch
from torch.utils.data import DataLoader

from .packed_data import PackedDataset, build_or_load_packed_cache
from .packed_datasets import MappedPackedDataset, SyntheticPackedDataset


def cache_paths(paths, tokenizer, settings):
    if settings.data_mode != "real":
        return None, None
    train_src, val_src = train_source(paths), val_source(paths)
    train_cache = build_or_load_packed_cache(paths, tokenizer, train_src, "train", settings)
    val_cache = build_or_load_packed_cache(paths, tokenizer, val_src, "val", settings)
    return train_cache, val_cache


def train_source(paths):
    if paths.public_pretrain_train.exists() and any(paths.public_pretrain_train.rglob("*.jsonl")):
        return paths.public_pretrain_train
    if paths.train_dataset.exists():
        return paths.train_dataset
    if paths.committed_train.exists() and any(paths.committed_train.rglob("*.jsonl")):
        return paths.committed_train
    return paths.fixtures


def val_source(paths):
    if paths.public_pretrain_val.exists() and any(paths.public_pretrain_val.rglob("*.jsonl")):
        return paths.public_pretrain_val
    if paths.val_dataset.exists():
        return paths.val_dataset
    if paths.committed_val.exists() and any(paths.committed_val.rglob("*.jsonl")):
        return paths.committed_val
    return paths.fixtures


def loader(cache, settings, pad_id: int, shuffle: bool, device, vocab_size: int, split: str):
    if settings.data_mode == "synthetic_gpu" and device.type == "cuda":
        return SyntheticGpuLoader(settings, vocab_size, split, device)
    if settings.data_mode == "synthetic_cpu":
        dataset = SyntheticPackedDataset(synthetic_windows(settings, split), settings.sequence_len, vocab_size, settings.seed)
        return DataLoader(dataset, batch_size=settings.batch_size, shuffle=shuffle, pin_memory=device.type == "cuda")
    dataset_cls = MappedPackedDataset if settings.dataloader_impl == "mapped" else PackedDataset
    kwargs = dataloader_kwargs(settings, shuffle, device, split)
    return DataLoader(dataset_cls(cache, settings.sequence_len, pad_id), **kwargs)


def dataloader_kwargs(settings, shuffle: bool, device, split: str) -> dict:
    kwargs = {
        "batch_size": settings.batch_size,
        "shuffle": shuffle,
        "pin_memory": device.type == "cuda" and settings.pin_memory,
        "num_workers": settings.num_workers,
        "drop_last": split == "train" and settings.static_shapes,
    }
    if settings.num_workers > 0:
        kwargs["prefetch_factor"] = settings.prefetch_factor
        kwargs["persistent_workers"] = settings.persistent_workers
    return kwargs


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
