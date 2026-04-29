import json
from pathlib import Path

from .tokenizer_tokens import BASE_SPECIAL_TOKENS

SPECIAL_TOKENS = BASE_SPECIAL_TOKENS


def message_text(messages: list[dict]) -> str:
    parts = dialogue_parts(messages)
    parts.extend(["</dialogue>", "<eos>"])
    return "\n".join(parts)


def prompt_text(messages: list[dict]) -> str:
    parts = dialogue_parts(messages)
    parts.append("<assistant_action>")
    return "\n".join(parts) + "\n"


def dialogue_parts(messages: list[dict]) -> list[str]:
    parts = ["<bos>", "<dialogue>"]
    for message in messages:
        if message["role"] == "assistant":
            parts.extend(["<assistant_action>", message["content"]])
            continue
        parts.extend(open_message(message))
        parts.append(escape_text(message["content"]))
        parts.extend(close_message())
    return parts


def row_text(row: dict) -> str:
    if isinstance(row.get("text"), str):
        return row["text"]
    return message_text(row.get("messages", []))


def supervised_token_ids(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    labels: list[int] = []
    for text, train in supervised_segments(row.get("messages", [])):
        encoded = tokenizer.encode(text).ids
        ids.extend(encoded)
        labels.extend(encoded if train else [-100] * len(encoded))
    return ids, labels


def supervised_segments(messages: list[dict]) -> list[tuple[str, bool]]:
    items: list[tuple[str, bool]] = [("<bos>", False), ("<dialogue>", False)]
    for message in messages:
        if message["role"] == "assistant":
            items.append(("<assistant_action>", False))
            items.append((message["content"], True))
            continue
        for segment in open_message(message):
            items.append((segment, False))
        items.append((escape_text(message["content"]), False))
        for segment in close_message():
            items.append((segment, False))
    items.extend([("</dialogue>", False), ("<eos>", False)])
    return [(text + ("\n" if index + 1 < len(items) else ""), train) for index, (text, train) in enumerate(items)]


def open_message(message: dict) -> list[str]:
    role = message["role"]
    name = message.get("name", "")
    parts = ["<message>", f"<role>{escape_text(role)}</role>"]
    if role == "tool" and name:
        parts.append(f"<tool_name>{escape_text(name)}</tool_name>")
    parts.append("<content>")
    return parts


def close_message() -> list[str]:
    return ["</content>", "</message>"]


def escape_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def load_rows(path) -> list[dict]:
    return list(iter_rows(path))


def iter_rows(path):
    path = Path(path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
