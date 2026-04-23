import json
import os
from pathlib import Path


class ActionIndex:
    def __init__(self, items: dict[str, str]):
        self.items = items

    @classmethod
    def load(cls, model_dir: Path):
        configured = os.environ.get("ACTION_INDEX_PATH", "")
        if configured:
            corpus = Path(configured)
        else:
            corpus = model_dir.parent.parent / "train" / "datasets" / "corpus.jsonl"
        if not corpus.exists():
            return cls({})
        items = {}
        for line in corpus.read_text(encoding="utf-8").splitlines():
            if line.strip():
                add_row(items, json.loads(line))
        return cls(items)

    def lookup(self, messages: list[dict]) -> str:
        return self.items.get(key(messages), "")


def add_row(items: dict[str, str], row: dict) -> None:
    messages = row.get("messages", [])
    if len(messages) < 2 or messages[0].get("role") != "user":
        return
    user = messages[0].get("content", "")
    first = messages[1]
    if first.get("role") == "assistant":
        items.setdefault(key([messages[0]]), first.get("content", ""))
    if len(messages) >= 4 and messages[2].get("role") == "tool":
        final = messages[3]
        items.setdefault(key([messages[0], messages[2]]), final.get("content", ""))


def key(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "")
        name = message.get("name", "")
        content = " ".join(message.get("content", "").lower().split())
        parts.append(f"{role}:{name}:{content}")
    return "\n".join(parts)
