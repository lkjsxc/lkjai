from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{20,}|[A-Za-z0-9_-]{32,}")


def load_api_keys(path: str = "") -> list[str]:
    values: list[str] = []
    for key in ["MOONSHOT_API_KEY", "KIMI_API_KEY"]:
        if os.environ.get(key):
            values.append(os.environ[key])
    if os.environ.get("MOONSHOT_API_KEYS"):
        values.extend(split_values(os.environ["MOONSHOT_API_KEYS"]))
    if path:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        values.extend(KEY_RE.findall(text))
    seen, keys = set(), []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)
    return keys


def split_values(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\s,;]+", text) if part.strip()]


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def redact(text: str, keys: list[str]) -> str:
    result = text
    for key in keys:
        if key:
            result = result.replace(key, f"<redacted:{fingerprint(key)}>")
    return result
