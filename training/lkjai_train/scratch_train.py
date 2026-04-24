import json
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset

from .formatting import load_rows, supervised_token_ids
from .scratch_model import ModelConfig, ScratchLM, parameter_count, save_config
from .tokenizer import load_tokenizer, train_tokenizer


def train_scratch(paths, settings) -> dict:
    paths.ensure()
    if not paths.tokenizer_json.exists():
        train_tokenizer(paths, settings)
    tokenizer = load_tokenizer(paths.tokenizer_json)
    train_rows = load_rows(paths.train_dataset if paths.train_dataset.exists() else paths.fixtures)
    val_rows = load_rows(paths.val_dataset if paths.val_dataset.exists() else paths.fixtures)
    train_windows = pack_rows(tokenizer, train_rows, settings.sequence_len)
    val_windows = pack_rows(tokenizer, val_rows or train_rows[:8], settings.sequence_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(settings.seed)
    config = model_config(settings, tokenizer.get_vocab_size())
    model = ScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(settings.max_steps))
    loader = DataLoader(TokenDataset(train_windows), batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(TokenDataset(val_windows), batch_size=settings.batch_size, shuffle=False)
    metrics = run_loop(model, optimizer, scheduler, loader, val_loader, settings, device)
    return save_training(paths, settings, config, model, len(train_rows), len(val_rows), metrics)


def pack_rows(tokenizer, rows: list[dict], sequence_len: int) -> list[tuple[list[int], list[int]]]:
    eos = tokenizer.token_to_id("<eos>") or 0
    pack_len = sequence_len + 1
    packed, current_ids, current_labels = [], [], []
    for row in rows:
        remaining_ids, remaining_labels = supervised_token_ids(tokenizer, row)
        remaining_ids.append(eos)
        remaining_labels.append(-100)
        while remaining_ids:
            space = pack_len - len(current_ids)
            current_ids.extend(remaining_ids[:space])
            current_labels.extend(remaining_labels[:space])
            remaining_ids = remaining_ids[space:]
            remaining_labels = remaining_labels[space:]
            if len(current_ids) == pack_len:
                packed.append((current_ids, current_labels))
                current_ids, current_labels = [], []
    if len(current_ids) > 1:
        pad = pack_len - len(current_ids)
        packed.append((current_ids + [eos] * pad, current_labels + [-100] * pad))
    return packed or [([eos] * pack_len, [-100] * pack_len)]


class TokenDataset(Dataset):
    def __init__(self, windows: list[tuple[list[int], list[int]]]):
        self.windows = [(torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)) for ids, labels in windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ids, labels = self.windows[index]
        return ids[:-1], labels[1:]


def model_config(settings, vocab_size: int) -> ModelConfig:
    return ModelConfig(vocab_size, settings.sequence_len, settings.layers, settings.hidden_size, settings.heads, settings.kv_heads, settings.ffn_size, 0.0)


def lr_lambda(max_steps: int):
    warmup = min(100, max(1, max_steps // 10))
    return lambda step: max(0.1, (step + 1) / warmup) if step < warmup else max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * (step - warmup) / max(1, max_steps - warmup))))


def run_loop(model, optimizer, scheduler, loader, val_loader, settings, device: torch.device) -> dict:
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    context = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    model.train()
    losses, best = [], float("inf")
    optimizer.zero_grad(set_to_none=True)
    step = 0
    while step < settings.max_steps:
        for input_ids, labels in loader:
            with context:
                _, loss, _ = model(input_ids.to(device), labels.to(device))
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
    val_loss = evaluate_loss(model, val_loader, device)
    return {"train_loss": losses[-1], "mean_loss": sum(losses) / len(losses), "best_train_loss": best, "val_loss": val_loss, "steps": step}


@torch.inference_mode()
def evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for index, (input_ids, labels) in enumerate(loader):
        if index >= 8:
            break
        _, loss, _ = model(input_ids.to(device), labels.to(device))
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_training(paths, settings, config, model, train_rows: int, val_rows: int, metrics: dict) -> dict:
    paths.checkpoint_final.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_best.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_final / "model.pt")
    torch.save(model.state_dict(), paths.checkpoint_best / "model.pt")
    save_config(config, paths.checkpoint_final / "config.json")
    save_config(config, paths.checkpoint_best / "config.json")
    summary = {"backend": "tiny-pytorch-scratch-v2", "checkpoint_dir": str(paths.checkpoint_final), "train_rows": train_rows, "val_rows": val_rows, "parameter_count": parameter_count(model), "settings": settings.model_preset, "metrics": metrics}
    paths.training_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
