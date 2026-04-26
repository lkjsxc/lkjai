import json
import os
from pathlib import Path
from typing import Any


class SettingsResolver:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def str(self, env_name: str, key: str, default: str) -> str:
        return str(self.value(env_name, key, default))

    def int(self, env_name: str, key: str, default: int) -> int:
        return int(self.value(env_name, key, default))

    def float(self, env_name: str, key: str, default: float) -> float:
        return float(self.value(env_name, key, default))

    def bool(self, env_name: str, key: str, default: bool) -> bool:
        value = self.value(env_name, key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def value(self, env_name: str, key: str, default):
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value.strip() != "":
            return env_value
        if key in self.config:
            return self.config[key]
        return default


def resolver_for(preset: str) -> SettingsResolver:
    if preset == "quick":
        return SettingsResolver({})
    path = os.environ.get("TRAIN_CONFIG", "").strip()
    if not path:
        return SettingsResolver({})
    return SettingsResolver(load_train_config(Path(path)))


def load_train_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("TRAIN_CONFIG must point to a JSON object")
    nested = data.get("settings")
    if nested is None:
        return data
    if not isinstance(nested, dict):
        raise ValueError("TRAIN_CONFIG settings must be a JSON object")
    merged = dict(data)
    merged.update(nested)
    return merged
