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
    train_ids, val_ids = split_ids(ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(settings.seed)
    config = model_config(settings, tokenizer.get_vocab_size())
    model = ScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(settings.max_steps))
    data = TokenDataset(train_ids, settings.sequence_len)
    loader = DataLoader(data, batch_size=settings.batch_size, shuffle=True)
    val = TokenDataset(val_ids, settings.sequence_len)
    metrics = run_loop(model, optimizer, scheduler, loader, val, settings, device)
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


def split_ids(ids: list[int]) -> tuple[list[int], list[int]]:
    cut = max(1, int(len(ids) * 0.9))
    return ids[:cut], ids[cut:] or ids[: max(2, len(ids) // 10)]


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


def lr_lambda(max_steps: int):
    warmup = min(100, max(1, max_steps // 10))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(0.1, (step + 1) / warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    return schedule


def run_loop(model, optimizer, scheduler, loader, val, settings, device: str) -> dict:
    model.train()
    step = 0
    losses = []
    best = float("inf")
    while step < settings.max_steps:
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(input_ids, labels)
            (loss / settings.gradient_accumulation).backward()
            if (step + 1) % settings.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            best = min(best, losses[-1])
            step += 1
            if step == 1 or step % 250 == 0:
                print(json.dumps({"event": "train_step", "step": step, "loss": losses[-1]}), flush=True)
            if step >= settings.max_steps:
                break
    if step % settings.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    val_loss = evaluate_loss(model, val, settings, device)
    return {
        "train_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "best_train_loss": best,
        "val_loss": val_loss,
        "steps": step,
    }


@torch.inference_mode()
def evaluate_loss(model, dataset, settings, device: str) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)
    losses = []
    for index, (input_ids, labels) in enumerate(loader):
        if index >= 8:
            break
        _, loss = model(input_ids.to(device), labels.to(device))
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_training(paths, settings, config, model, rows, metrics: dict) -> dict:
    paths.checkpoint_final.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_best.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_final / "model.pt")
    torch.save(model.state_dict(), paths.checkpoint_best / "model.pt")
    save_config(config, paths.checkpoint_final / "config.json")
    save_config(config, paths.checkpoint_best / "config.json")
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
