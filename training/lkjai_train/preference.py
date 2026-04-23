import json
import os
from pathlib import Path

import torch

from .behavioral import CASES
from .formatting import prompt_text
from .scratch_model import ModelConfig, ScratchLM, save_config
from .tokenizer import load_tokenizer


def prepare_preferences(paths) -> Path:
    paths.ensure()
    pairs = behavioral_pairs() + synthetic_pairs()
    with paths.preference_pairs.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return paths.preference_pairs


def behavioral_pairs() -> list[dict]:
    pairs = []
    for case in CASES:
        chosen = expected_action(case)
        rejected = {"kind": "final", "content": "I cannot help."}
        pairs.append(pair(case["messages"], chosen, rejected, f"behavioral:{case['id']}"))
    return pairs


def synthetic_pairs() -> list[dict]:
    return [
        pair(
            [{"role": "user", "content": "Remember that my favorite editor is vim."}],
            {"kind": "tool_call", "tool": "memory.write", "args": {"content": "User favorite editor is vim."}},
            {"kind": "final", "content": "Vim is a text editor."},
            "memory-tool-selection",
        ),
        pair(
            [{"role": "user", "content": "Read /etc/passwd."}],
            {"kind": "final", "content": "I can only use files inside the data workspace."},
            {"kind": "tool_call", "tool": "fs.read", "args": {"path": "/etc/passwd"}},
            "workspace-boundary",
        ),
    ]


def expected_action(case: dict) -> dict:
    if case["kind"] == "tool_call":
        return {"kind": "tool_call", "tool": case["tool"], "args": default_args(case["tool"])}
    content = case.get("contains", "ok")
    return {"kind": "final", "content": str(content)}


def default_args(tool: str) -> dict:
    return {
        "fs.list": {"path": "."},
        "fs.read": {"path": "README.md"},
        "fs.write": {"path": "notes.txt", "content": "note"},
        "shell.exec": {"command": "pwd"},
        "web.fetch": {"url": "https://example.com"},
        "memory.write": {"content": "User prefers concise answers."},
        "memory.search": {"query": "preferences"},
    }.get(tool, {})


def pair(messages: list[dict], chosen: dict, rejected: dict, source: str) -> dict:
    return {
        "messages": messages,
        "chosen": json.dumps(chosen, ensure_ascii=False),
        "rejected": json.dumps(rejected, ensure_ascii=False),
        "source": source,
    }


def train_dpo(paths, settings) -> Path:
    if not paths.preference_pairs.exists():
        prepare_preferences(paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(paths.tokenizer_json)
    config = ModelConfig(**json.loads((paths.checkpoint_final / "config.json").read_text()))
    model = ScratchLM(config).to(device)
    state = torch.load(paths.checkpoint_final / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate * 0.25)
    pairs = read_pairs(paths.preference_pairs)
    steps = int(os.environ.get("TRAIN_DPO_STEPS", str(min(200, max(20, settings.max_steps // 10)))))
    beta = float(os.environ.get("TRAIN_DPO_BETA", "0.1"))
    losses = []
    for step in range(steps):
        item = pairs[step % len(pairs)]
        chosen = sequence_logp(model, tokenizer, config, item["messages"], item["chosen"], device)
        rejected = sequence_logp(model, tokenizer, config, item["messages"], item["rejected"], device)
        loss = -torch.nn.functional.logsigmoid(beta * (chosen - rejected))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    save_dpo(paths, config, model, {"dpo_loss": losses[-1], "mean_dpo_loss": sum(losses) / len(losses), "steps": steps})
    return paths.dpo_summary


def sequence_logp(model, tokenizer, config: ModelConfig, messages, response: str, device: str):
    prompt_ids = tokenizer.encode(prompt_text(messages)).ids
    response_ids = tokenizer.encode(response + "\n<eos>").ids
    ids = (prompt_ids + response_ids)[-config.sequence_len :]
    prompt_len = max(1, min(len(prompt_ids), len(ids) - 1))
    input_ids = torch.tensor([ids[:-1]], device=device)
    targets = torch.tensor([ids[1:]], device=device)
    logits, _ = model(input_ids)
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp[:, prompt_len - 1 :].sum()


def read_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_dpo(paths, config, model, metrics: dict) -> None:
    paths.checkpoint_dpo.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_dpo / "model.pt")
    save_config(config, paths.checkpoint_dpo / "config.json")
    summary = {"backend": "dpo-lite", "accepted": True, "metrics": metrics}
    paths.dpo_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def mark_dpo_rejected(paths, reason: str) -> None:
    if not paths.dpo_summary.exists():
        return
    data = json.loads(paths.dpo_summary.read_text(encoding="utf-8"))
    data["accepted"] = False
    data["rejection_reason"] = reason
    paths.dpo_summary.write_text(json.dumps(data, indent=2), encoding="utf-8")
