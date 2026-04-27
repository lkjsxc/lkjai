import json
import os
from pathlib import Path

from .public_pretrain import public_pretrain_sources, validate_source


def download_public_pretrain(paths) -> Path:
    sources = public_pretrain_sources()
    for source in sources:
        validate_source(source)
    if not sources:
        raise ValueError("no active public pretrain sources")
    dataset = shared_value(sources, "dataset")
    revision = shared_value(sources, "revision")
    target = Path(os.environ.get("TRAIN_PUBLIC_DATA_DIR", str(paths.root / "raw" / "cosmopedia")))
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(dataset, revision, target, allow_patterns(sources), hf_token())
    report = {
        "schema": "lkjai-public-pretrain-download-v1",
        "dataset": dataset,
        "revision": revision,
        "target_dir": str(target),
        "allow_patterns": allow_patterns(sources),
        "source_count": len(sources),
        "token_source": token_source(),
    }
    paths.public_pretrain.mkdir(parents=True, exist_ok=True)
    path = paths.public_pretrain / "download-report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def snapshot_download(dataset: str, revision: str, target: Path, patterns: list[str], token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub or use the train Docker image") from exc
    kwargs = {
        "repo_id": dataset,
        "repo_type": "dataset",
        "revision": revision,
        "local_dir": str(target),
        "allow_patterns": patterns,
        "token": token,
    }
    try:
        hf_snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        hf_snapshot_download(**kwargs)


def allow_patterns(sources: list[dict]) -> list[str]:
    patterns = {source["local_glob"] for source in sources}
    patterns.add("README.md")
    return sorted(patterns)


def shared_value(sources: list[dict], key: str) -> str:
    values = {str(source[key]) for source in sources}
    if len(values) != 1:
        raise ValueError(f"public pretrain sources must share {key}")
    return values.pop()


def hf_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token.strip()
    path = os.environ.get("HF_TOKEN_FILE", "").strip()
    if not path:
        return None
    return first_token(Path(path).read_text(encoding="utf-8"))


def first_token(text: str) -> str | None:
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if "hf_" in value:
            return value[value.index("hf_") :].split()[0].strip()
        return value.split()[-1]
    return None


def token_source() -> str:
    if os.environ.get("HF_TOKEN"):
        return "HF_TOKEN"
    if os.environ.get("HF_TOKEN_FILE"):
        return "HF_TOKEN_FILE"
    return "none"
