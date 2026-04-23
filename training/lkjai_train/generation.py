import json
from pathlib import Path

import torch

from .formatting import prompt_text
from .scratch_model import ModelConfig, ScratchLM
from .tokenizer import load_tokenizer


class LoadedModel:
    def __init__(self, model_dir: Path, device: str = "") -> None:
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_dir / "tokenizer.json")
        config = ModelConfig(**json.loads((model_dir / "config.json").read_text()))
        self.model = ScratchLM(config).to(self.device)
        state = torch.load(model_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.config = config

    @torch.inference_mode()
    def complete(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.0) -> str:
        encoded = self.tokenizer.encode(prompt_text(messages)).ids
        input_ids = torch.tensor([encoded[-self.config.sequence_len :]], device=self.device)
        eos = self.tokenizer.token_to_id("<eos>")
        for _ in range(max_tokens):
            window = input_ids[:, -self.config.sequence_len :]
            logits, _ = self.model(window)
            next_logits = logits[:, -1, :]
            next_id = choose_token(next_logits, temperature)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos is not None and int(next_id.item()) == eos:
                break
        new_ids = input_ids[0, len(encoded) :].detach().cpu().tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        return normalize_action(text)


def choose_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def normalize_action(text: str) -> str:
    candidate = first_json_object(text)
    if candidate:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    cleaned = text.replace("<eos>", "").strip()
    return json.dumps({
        "kind": "final",
        "thought": "generated text did not contain valid action json",
        "content": cleaned,
    })


def first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""
