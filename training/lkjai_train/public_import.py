import json
import os
from pathlib import Path

from .corpus_source import tagged_contents
from .rows import row


ALLOWED_LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"}


def public_sources() -> list[dict]:
    return tagged_contents("public", "public_dataset")


def validate_public_sources(paths) -> Path:
    paths.ensure()
    sources = public_sources()
    for source in sources:
        license_name = source.get("license", "")
        if license_name not in ALLOWED_LICENSES:
            raise ValueError(f"public source {source.get('name')} has disallowed license {license_name}")
        for key in ["name", "license", "source_url", "revision", "local_file"]:
            if not source.get(key):
                raise ValueError(f"public source missing {key}")
    paths.public_manifest.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")
    return paths.public_manifest


def prepare_public_corpus(paths) -> Path:
    validate_public_sources(paths)
    root_value = os.environ.get("TRAIN_PUBLIC_DATA_DIR", "")
    root = Path(root_value) if root_value else None
    rows = []
    for source in public_sources():
        local = root / source["local_file"] if root else None
        if local and local.is_file():
            rows.extend(read_public_rows(local, source))
    with paths.public_corpus.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    update_manifest(paths, len(rows))
    return paths.public_corpus


def read_public_rows(path: Path, source: dict) -> list[dict]:
    limit = int(source.get("row_limit", 0))
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if limit and len(rows) >= limit:
            break
        item = json.loads(line)
        messages = item.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        rows.append(row(messages, source.get("tags", []), meta(source, index)))
    return rows


def meta(source: dict, index: int) -> dict:
    return {
        "id": f"public-{source['name']}-{index:06d}",
        "split": "train",
        "provenance": "public-import",
        "author_type": "external",
        "author_model": "unknown",
        "quality_tier": "medium",
        "domain": source["name"],
        "skill": source.get("skill", "instruction"),
        "toolset": source.get("toolset", "none"),
        "language": source.get("language", "en"),
        "safety_scope": "workspace-safe",
        "license": source["license"],
        "source_ref": f"{source['source_url']}@{source['revision']}",
    }


def update_manifest(paths, imported_rows: int) -> None:
    data = json.loads(paths.public_manifest.read_text(encoding="utf-8"))
    data["imported_rows"] = imported_rows
    paths.public_manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
