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
    parts = ["<bos>", "<conversation>"]
    for message in messages:
        parts.append(open_message(message))
        parts.append(message["content"])
        parts.append("</message>")
    parts.extend(["</conversation>", "<eos>"])
    return "\n".join(parts)


def prompt_text(messages: list[dict]) -> str:
    parts = ["<bos>", "<conversation>"]
    for message in messages:
        parts.append(open_message(message))
        parts.append(message["content"])
        parts.append("</message>")
    parts.extend(["</conversation>", '<message role="assistant">'])
    return "\n".join(parts)


def row_text(row: dict) -> str:
    return message_text(row.get("messages", []))


def open_message(message: dict) -> str:
    role = message["role"]
    name = message.get("name", "")
    if name:
        return f'<message role="{role}" name="{name}">'
    return f'<message role="{role}">'


def load_rows(path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
