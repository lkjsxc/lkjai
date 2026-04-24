import json
import os
from pathlib import Path

from .dataset import prepare_fixtures
from .formatting import load_rows, prompt_text


def prepare_preferences(paths) -> Path:
    paths.ensure()
    if not paths.holdout_dataset.exists() and not paths.fixtures.exists():
        prepare_fixtures(paths)
    rows = load_rows(paths.holdout_dataset if paths.holdout_dataset.exists() else paths.fixtures)
    pairs = [row_pair(row) for row in rows if row.get("messages", []) and row["messages"][-1]["role"] == "assistant"]
    with paths.preference_pairs.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return paths.preference_pairs


def row_pair(row: dict) -> dict:
    chosen = row["messages"][-1]["content"]
    expected = json.loads(chosen)
    rejected = rejection_for(expected)
    return {
        "messages": row["messages"][:-1],
        "chosen": chosen,
        "rejected": json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
        "source": row["meta"]["id"],
    }


def rejection_for(expected: dict) -> dict:
    if expected.get("kind") == "tool_call":
        return {"kind": "final", "content": "done"}
    if expected.get("kind") == "request_confirmation":
        pending = expected.get("pending_tool_call", {})
        return {"kind": "tool_call", "tool": pending.get("tool", "resource.update_resource"), "args": pending.get("args", {})}
    return {"kind": "final", "content": "I cannot help."}


def train_dpo(paths, settings) -> Path:
    import torch

    from .scratch_model import ModelConfig, ScratchLM, save_config
    from .tokenizer import load_tokenizer

    if not paths.preference_pairs.exists():
        prepare_preferences(paths)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(paths.tokenizer_json)
    config = ModelConfig(**json.loads((paths.checkpoint_final / "config.json").read_text()))
    model = load_model(config, paths.checkpoint_final / "model.pt", device)
    reference = load_model(config, paths.checkpoint_final / "model.pt", device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate * 0.25)
    pairs = read_pairs(paths.preference_pairs)
    steps = int(os.environ.get("TRAIN_DPO_STEPS", str(min(200, max(20, settings.max_steps // 10)))))
    beta = float(os.environ.get("TRAIN_DPO_BETA", "0.1"))
    losses = run_dpo_steps(model, reference, tokenizer, config, pairs, steps, beta, optimizer, device)
    save_dpo(paths, config, model, {"dpo_loss": losses[-1], "mean_dpo_loss": sum(losses) / len(losses), "steps": steps, "beta": beta, "reference": str(paths.checkpoint_final)})
    return paths.dpo_summary


def load_model(config, path: Path, device: str):
    import torch

    from .scratch_model import ScratchLM

    model = ScratchLM(config).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def run_dpo_steps(model, reference, tokenizer, config, pairs, steps, beta, optimizer, device) -> list[float]:
    import torch

    losses = []
    for step in range(steps):
        item = pairs[step % len(pairs)]
        chosen = sequence_logp(model, tokenizer, config, item["messages"], item["chosen"], device)
        rejected = sequence_logp(model, tokenizer, config, item["messages"], item["rejected"], device)
        with torch.no_grad():
            ref_chosen = sequence_logp(reference, tokenizer, config, item["messages"], item["chosen"], device)
            ref_rejected = sequence_logp(reference, tokenizer, config, item["messages"], item["rejected"], device)
        policy_margin = chosen - rejected
        reference_margin = ref_chosen - ref_rejected
        loss = -torch.nn.functional.logsigmoid(beta * (policy_margin - reference_margin))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    return losses


def sequence_logp(model, tokenizer, config, messages, response: str, device: str):
    import torch

    prompt_ids = tokenizer.encode(prompt_text(messages)).ids
    response_ids = tokenizer.encode(response + "\n<eos>").ids
    ids = (prompt_ids + response_ids)[-config.sequence_len :]
    prompt_len = max(1, min(len(prompt_ids), len(ids) - 1))
    input_ids = torch.tensor([ids[:-1]], device=device)
    targets = torch.tensor([ids[1:]], device=device)
    logits, _, _ = model(input_ids)
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp[:, prompt_len - 1 :].sum()


def read_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_dpo(paths, config, model, metrics: dict) -> None:
    import torch

    from .scratch_model import save_config

    paths.checkpoint_dpo.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), paths.checkpoint_dpo / "model.pt")
    save_config(config, paths.checkpoint_dpo / "config.json")
    summary = {"backend": "dpo-lite", "accepted": False, "pending_evaluation": True, "metrics": metrics}
    paths.dpo_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
