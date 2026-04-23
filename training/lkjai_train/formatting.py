import json


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<assistant_json>",
]


def message_text(messages: list[dict]) -> str:
    parts = ["<bos>", "<dialogue>"]
    for message in messages:
        if message["role"] == "assistant":
            parts.extend(["<assistant_json>", message["content"]])
            continue
        parts.append(open_message(message))
        parts.append(escape_text(message["content"]))
        parts.append(close_message())
    parts.extend(["</dialogue>", "<eos>"])
    return "\n".join(parts)


def prompt_text(messages: list[dict]) -> str:
    parts = ["<bos>", "<dialogue>"]
    for message in messages:
        if message["role"] == "assistant":
            parts.extend(["<assistant_json>", message["content"]])
            continue
        parts.append(open_message(message))
        parts.append(escape_text(message["content"]))
        parts.append(close_message())
    parts.extend(["</dialogue>", "<assistant_json>"])
    return "\n".join(parts)


def row_text(row: dict) -> str:
    return message_text(row.get("messages", []))


def open_message(message: dict) -> str:
    role = message["role"]
    name = message.get("name", "")
    if role == "tool" and name:
        return f'<message role="{role}"><tool_name>{escape_text(name)}</tool_name><content>'
    return f'<message role="{role}"><content>'


def close_message() -> str:
    return "</content></message>"


def escape_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def load_rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
