import json


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
]


def message_text(messages: list[dict]) -> str:
    parts = ["<bos>", "<conversation>"]
    for message in messages:
        parts.append(open_message(message))
        parts.append(message["content"])
        parts.append(close_message(message))
    parts.extend(["</conversation>", "<eos>"])
    return "\n".join(parts)


def prompt_text(messages: list[dict]) -> str:
    parts = ["<bos>", "<conversation>"]
    for message in messages:
        parts.append(open_message(message))
        parts.append(message["content"])
        parts.append(close_message(message))
    parts.extend(["</conversation>", "<assistant>"])
    return "\n".join(parts)


def row_text(row: dict) -> str:
    return message_text(row.get("messages", []))


def open_message(message: dict) -> str:
    role = message["role"]
    name = message.get("name", "")
    if role == "tool" and name:
        return f"<tool>{name}\n"
    return f"<{role}>"


def close_message(message: dict) -> str:
    role = message["role"]
    return f"</{role}>"


def load_rows(path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows
