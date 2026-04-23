import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .formatting import load_rows, row_text
from .scratch_model import ModelConfig, ScratchLM, parameter_count, save_config
from .tokenizer import load_tokenizer, train_tokenizer


def train_scratch(paths, settings) -> dict:
    paths.ensure()
    dataset_path = resolve_dataset(paths)
    if not paths.tokenizer_json.exists():
        train_tokenizer(paths, settings)
    tokenizer = load_tokenizer(paths.tokenizer_json)
    rows = load_rows(dataset_path)
    ids = encode_rows(tokenizer, rows)
    if len(ids) < settings.sequence_len + 2:
        ids = ids * ((settings.sequence_len + 2) // max(1, len(ids)) + 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(settings.seed)
    config = model_config(settings, tokenizer.get_vocab_size())
    model = ScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    data = TokenDataset(ids, settings.sequence_len)
    loader = DataLoader(data, batch_size=settings.batch_size, shuffle=True)
    metrics = run_loop(model, optimizer, loader, settings, device)
    return save_training(paths, settings, config, model, rows, metrics)


def resolve_dataset(paths) -> Path:
    env_path = PathEnv.get("TRAIN_DATASET_PATH")
    if env_path:
        return Path(env_path)
    return paths.corpus if paths.corpus.exists() else paths.fixtures


class PathEnv:
    @staticmethod
    def get(key: str) -> str:
        import os

        return os.environ.get(key, "")


def encode_rows(tokenizer, rows: list[dict]) -> list[int]:
    eos = tokenizer.token_to_id("<eos>")
    ids = []
    for row in rows:
        ids.extend(tokenizer.encode(row_text(row)).ids)
        if eos is not None:
            ids.append(eos)
    return ids


class TokenDataset(Dataset):
    def __init__(self, ids: list[int], sequence_len: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.sequence_len = sequence_len
        self.count = max(1, len(ids) - sequence_len - 1)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index % self.count
        item = self.ids[start : start + self.sequence_len]
        return item, item.clone()


def model_config(settings, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        sequence_len=settings.sequence_len,
        layers=settings.layers,
        hidden_size=settings.hidden_size,
        heads=settings.heads,
        kv_heads=settings.kv_heads,
        ffn_size=settings.ffn_size,
        dropout=0.0,
    )


def run_loop(model, optimizer, loader, settings, device: str) -> dict:
    model.train()
    step = 0
    losses = []
    while step < settings.max_steps:
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            _, loss = model(input_ids, labels)
            (loss / settings.gradient_accumulation).backward()
            if (step + 1) % settings.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step >= settings.max_steps:
                break
    if step % settings.gradient_accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": losses[-1], "mean_loss": sum(losses) / len(losses), "steps": step}


def save_training(paths, settings, config, model, rows, metrics: dict) -> dict:
    paths.checkpoint_final.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_final / "model.pt")
    save_config(config, paths.checkpoint_final / "config.json")
    summary = {
        "backend": "tiny-pytorch-scratch",
        "checkpoint_dir": str(paths.checkpoint_final),
        "train_rows": len(rows),
        "parameter_count": parameter_count(model),
        "settings": settings.model_preset,
        "metrics": metrics,
    }
    paths.training_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
