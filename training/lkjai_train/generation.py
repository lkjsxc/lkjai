import json
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree

from .formatting import prompt_text


def choose_token(logits, temperature: float):
    import torch

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, num_samples=1)


class LoadedModel:
    def __init__(self, model_dir: Path, device: str = "") -> None:
        import torch

        from .scratch_model import ModelConfig, ScratchLM
        from .tokenizer import load_tokenizer

        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = load_tokenizer(model_dir / "tokenizer.json")
        config = ModelConfig(**json.loads((model_dir / "config.json").read_text()))
        self.model = ScratchLM(config).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        self.config = config

    def complete(self, messages: list[dict], max_tokens: int = 128, temperature: float = 0.0) -> str:
        import torch

        normalized = normalize_messages(messages)
        prompt_ids = self.tokenizer.encode(prompt_text(normalized)).ids[-self.config.sequence_len :]
        input_ids = torch.tensor([prompt_ids], device=self.device)
        generated = []
        eos = self.tokenizer.token_to_id("<eos>")
        with torch.inference_mode():
            context = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
            with context:
                logits, _, cache = self.model(input_ids, use_cache=True)
                next_logits = logits[:, -1, :]
                for _ in range(max_tokens):
                    next_id = choose_token(next_logits, temperature)
                    token = int(next_id.item())
                    generated.append(token)
                    if eos is not None and token == eos:
                        break
                    text = self.tokenizer.decode(generated, skip_special_tokens=False)
                    if "</action>" in text:
                        break
                    logits, _, cache = self.model(next_id, cache=cache, use_cache=True)
                    next_logits = logits[:, -1, :]
        return normalize_action(self.tokenizer.decode(generated, skip_special_tokens=False))


def normalize_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    latest = messages[-1].get("content", "")
    agent_messages = agent_context_messages(latest)
    if agent_messages:
        return [normalize_message(message) for message in agent_messages]
    extracted = latest_user_event(latest)
    if extracted:
        return [normalize_message({"role": "user", "content": extracted})]
    return [normalize_message(message) for message in messages]


def agent_context_messages(content: str) -> list[dict]:
    if "<events>" in content:
        user = latest_tagged_event(content, "user")
        observation = latest_tagged_event(content, "observation")
        if user and (observation or has_tagged_event(content, "observation")):
            return [{"role": "user", "content": user}, {"role": "tool", "name": guessed_tool(user), "content": observation}]
        return [{"role": "user", "content": user}] if user else []
    return []


def latest_user_event(content: str) -> str:
    return latest_tagged_event(content, "user") if "<events>" in content else ""


def latest_tagged_event(content: str, kind: str) -> str:
    parsed = latest_structured_event(content, kind)
    if parsed:
        return parsed
    marker, end = f'<event kind="{kind}">', "</event>"
    index = content.rfind(marker)
    if index < 0:
        return ""
    start = index + len(marker)
    stop = content.find(end, start)
    return "" if stop < 0 else unescape_xml(content[start:stop]).strip()


def has_tagged_event(content: str, kind: str) -> bool:
    return f'<event kind="{kind}">' in content or f"<kind>{kind}</kind>" in content


def latest_structured_event(content: str, kind: str) -> str:
    try:
        root = ElementTree.fromstring(f"<root>{content}</root>")
    except ElementTree.ParseError:
        return ""
    matches = []
    for event in root.findall(".//event"):
        event_kind = event.findtext("kind", default="")
        if event_kind == kind:
            matches.append(event.findtext("content", default=""))
    return unescape_xml(matches[-1]).strip() if matches else ""


def guessed_tool(user: str) -> str:
    lower = user.lower()
    if "remember" in lower:
        return "memory.write"
    if "search" in lower and "resource" in lower:
        return "resource.search"
    if "preview" in lower:
        return "resource.preview_markdown"
    if "list" in lower:
        return "fs.list"
    return "tool"


def unescape_xml(text: str) -> str:
    return text.replace("&lt;", "<").replace("&amp;", "&")


def normalize_message(message: dict) -> dict:
    if message.get("role") != "user":
        return message
    content = message.get("content", "")
    if "<task>" in content:
        return message
    return {"role": "user", "content": default_task(content)}


def default_task(request: str) -> str:
    return (
        "<task>\n"
        f"<request>{request}</request>\n"
        "<context></context>\n"
        "<constraints>Return one valid XML action.</constraints>\n"
        "</task>"
    )


def normalize_action(text: str) -> str:
    candidate = first_xml_action(text)
    if candidate:
        return candidate
    text = text.replace("<eos>", "").replace("<assistant_action>", "").strip()
    if text.startswith("action>"):
        return f"<{text}"
    if text.startswith(("reasoning>", "tool>")):
        return f"<action>\n<{text}"
    return text


def first_xml_action(text: str) -> str:
    start = text.find("<action>")
    if start < 0:
        return ""
    end = text.find("</action>", start)
    if end >= 0:
        return text[start : end + len("</action>")]
    return ""
