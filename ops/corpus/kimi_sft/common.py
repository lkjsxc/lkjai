import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .constants import SECRET_RE, SPLITS


@dataclass
class ApiResult:
    status: str
    text: str = ""
    error: str = ""
    attempts: int = 0
    elapsed_ms: int = 0
    retryable: bool = False
    quota_exhausted: bool = False
    unauthorized: bool = False
    access_terminated: bool = False


def parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(item) for item in body.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def read_config(path):
    config = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise SystemExit(f"bad config line in {path}: {raw.rstrip()}")
            key, value = line.split(":", 1)
            config[key.strip()] = parse_scalar(value)
    apply_env_overrides(config)
    return config


def apply_env_overrides(config):
    overrides = {
        "KIMI_API_BASE_URL": "api_base_url",
        "KIMI_API_MODEL": "api_model",
        "KIMI_USER_AGENT": "user_agent",
        "KIMI_CLI_RUNNER": "kimi_cli_runner",
    }
    for env_name, config_name in overrides.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            config[config_name] = value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def load_api_key():
    key_file = os.environ.get("KIMI_API_KEY_FILE", "").strip()
    if key_file:
        path = Path(key_file)
        if not path.is_file():
            raise SystemExit("KIMI_API_KEY_FILE does not exist")
        body = path.read_text(encoding="utf-8").strip()
        matches = SECRET_RE.findall(body)
        if matches:
            return matches[0]
        if body:
            return body.splitlines()[0].strip()
    key = os.environ.get("KIMI_API_KEY", "").strip()
    if key:
        return key
    raise SystemExit("missing Kimi API key; set KIMI_API_KEY_FILE or KIMI_API_KEY")


def estimated_tokens_for_row(row):
    return max(1, len(json.dumps(row, ensure_ascii=False, separators=(",", ":"))) // 4)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(root.glob("*/*.jsonl")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(root):
    for split in SPLITS:
        for path in sorted((Path(root) / split).glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield split, path, number, json.loads(line), ""
                    except json.JSONDecodeError as exc:
                        yield split, path, number, None, f"bad JSONL: {exc.msg}"
