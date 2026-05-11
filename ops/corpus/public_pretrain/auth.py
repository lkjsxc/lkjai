import os
from pathlib import Path

from .common import DEFAULT_SECRET, TOKEN_RE


def token_from_markdown(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            in_section = stripped.lower() == "### huggingface, api-key"
            continue
        if in_section and stripped:
            return stripped
    return ""


def resolve_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token_file = os.environ.get("HF_TOKEN_FILE", DEFAULT_SECRET)
        if token_file and Path(token_file).is_file():
            raw = Path(token_file).read_text(encoding="utf-8")
            if "### huggingface, api-key" in raw.lower():
                token = token_from_markdown(token_file)
            else:
                token = raw.strip()
    if not TOKEN_RE.fullmatch(token):
        raise SystemExit("missing or malformed Hugging Face token")
    return token


def auth_headers():
    return {"Authorization": f"Bearer {resolve_token()}"}
