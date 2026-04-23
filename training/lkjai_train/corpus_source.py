import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parents[1] / "corpus_sources"


@lru_cache(maxsize=None)
def load_entries(name: str) -> tuple[dict, ...]:
    path = SOURCE_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain one JSON array")
    entries = tuple(validate_entry(path, index, item) for index, item in enumerate(data, start=1))
    return entries


def tagged_contents(name: str, tag: str) -> list[dict[str, Any]]:
    return [entry["content"] for entry in load_entries(name) if tag in entry["tags"]]


def texts(name: str, tag: str) -> list[str]:
    return [content["text"] for content in tagged_contents(name, tag)]


def validate_entry(path: Path, index: int, item: Any) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"{path}[{index}] must be an object")
    tags = item.get("tags")
    content = item.get("content")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag for tag in tags):
        raise ValueError(f"{path}[{index}].tags must be a non-empty string array")
    if not isinstance(content, dict):
        raise ValueError(f"{path}[{index}].content must be an object")
    return {"tags": tags, "content": content}


def validate_sources() -> None:
    for path in sorted(SOURCE_DIR.glob("*.json")):
        load_entries(path.stem)
