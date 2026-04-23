import json


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
]


def message_text(messages: list[dict]) -> str:
    parts = ["<bos>"]
    for message in messages:
        role = message["role"]
        name = message.get("name", "")
        header = f"<|{role}|>"
        if name:
            header = f"{header} {name}"
        parts.append(header)
        parts.append(message["content"])
    parts.append("<eos>")
    return "\n".join(parts)


def prompt_text(messages: list[dict]) -> str:
    parts = ["<bos>"]
    for message in messages:
        role = message["role"]
        name = message.get("name", "")
        header = f"<|{role}|>"
        if name:
            header = f"{header} {name}"
        parts.append(header)
        parts.append(message["content"])
    parts.append("<|assistant|>")
    return "\n".join(parts)


def row_text(row: dict) -> str:
    return message_text(row.get("messages", []))


def load_rows(path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
