import hashlib
import json
import os
import re
import time
from pathlib import Path

DEFAULT_SOURCE = "/workspace/corpus/sources/public-pretrain.json"
DEFAULT_RAW = "/app/data/raw/cosmopedia"
DEFAULT_OUT = "/app/data/public-corpus"
DEFAULT_SECRET = "/run/secrets/hf_token_source"
PARQUET_API = "https://datasets-server.huggingface.co/parquet"
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_]{20,}")


def env_path(name, default):
    return Path(os.environ.get(name, default))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def load_recipes(source_file):
    recipes = []
    for entry in read_json(source_file):
        content = entry.get("content", {})
        if content.get("approval_status") == "active":
            recipes.append(content)
    if not recipes:
        raise SystemExit("no active public-pretrain recipes")
    return recipes


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_command_meta(command):
    return {
        "command": command,
        "source_file": os.environ.get("CORPUS_SOURCE_FILE", DEFAULT_SOURCE),
        "raw_dir": os.environ.get("TRAIN_PUBLIC_DATA_DIR", DEFAULT_RAW),
        "out_dir": os.environ.get("TRAIN_CORPUS_DIR", DEFAULT_OUT),
        "target_tokens": int(os.environ.get("TRAIN_PUBLIC_PRETRAIN_TOKENS",
                                            "500000000")),
        "hf_home": os.environ.get("HF_HOME", ""),
        "token_source": "HF_TOKEN/HF_TOKEN_FILE",
        "created_unix": int(time.time()),
    }


def estimated_tokens(value, text):
    if value is None:
        return max(1, len(text) // 4)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, len(text) // 4)


def split_for_ordinal(ordinal):
    mod = ordinal % 100
    if mod == 0:
        return "val"
    if mod == 1:
        return "holdout"
    return "train"
