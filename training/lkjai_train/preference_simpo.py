import json
import os

import torch

from .preference import prepare_preferences, read_pairs, sequence_logp
from .scratch_model import ModelConfig, save_config
from .tokenizer import load_tokenizer


def train_simpo(paths, settings):
    if not paths.preference_pairs.exists():
        prepare_preferences(paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(paths.tokenizer_json)
    config = ModelConfig(**json.loads((paths.checkpoint_final / "config.json").read_text()))
    from .preference import load_model

    model = load_model(config, paths.checkpoint_final / "model.pt", device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate * 0.25)
    pairs = read_pairs(paths.preference_pairs)
    steps = int(os.environ.get("TRAIN_SIMPO_STEPS", str(min(200, max(20, settings.max_steps // 10)))))
    beta = float(os.environ.get("TRAIN_SIMPO_BETA", "2.0"))
    gamma = float(os.environ.get("TRAIN_SIMPO_GAMMA", "0.5"))
    losses = run_simpo_steps(model, tokenizer, config, pairs, steps, beta, gamma, optimizer, device)
    save_simpo(paths, config, model, {"simpo_loss": losses[-1], "mean_simpo_loss": sum(losses) / len(losses), "steps": steps, "beta": beta, "gamma": gamma})
    return paths.simpo_summary


def run_simpo_steps(model, tokenizer, config, pairs, steps, beta, gamma, optimizer, device) -> list[float]:
    losses = []
    for step in range(steps):
        item = pairs[step % len(pairs)]
        chosen = sequence_logp(model, tokenizer, config, item["messages"], item["chosen"], device, average=True)
        rejected = sequence_logp(model, tokenizer, config, item["messages"], item["rejected"], device, average=True)
        loss = -torch.nn.functional.logsigmoid(beta * (chosen - rejected - gamma))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return losses


def save_simpo(paths, config, model, metrics: dict) -> None:
    paths.checkpoint_simpo.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_simpo / "model.pt")
    save_config(config, paths.checkpoint_simpo / "config.json")
    summary = {"backend": "simpo", "accepted": False, "pending_evaluation": True, "metrics": metrics}
    paths.simpo_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
