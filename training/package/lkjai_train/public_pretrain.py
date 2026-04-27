import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .corpus_source import tagged_contents
from .formatting import row_text
from .public_import import ALLOWED_LICENSES, UNPINNED_REVISIONS
from .public_pretrain_readers import iter_source_text, source_paths


def public_pretrain_sources() -> list[dict]:
    return tagged_contents("public-pretrain", "public_pretrain_dataset")


def validate_public_pretrain_sources(paths) -> Path:
    paths.public_pretrain.mkdir(parents=True, exist_ok=True)
    sources = public_pretrain_sources()
    for source in sources:
        validate_source(source)
    if not sources:
        raise ValueError("no active public pretrain sources")
    paths.public_pretrain_manifest.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")
    return paths.public_pretrain_manifest


def validate_source(source: dict) -> None:
    required = ["name", "dataset", "config", "license", "source_url", "revision", "text_field", "language", "token_budget", "field_policy"]
    for key in required:
        if not source.get(key):
            raise ValueError(f"public pretrain source missing {key}")
    if not source.get("local_file") and not source.get("local_glob"):
        raise ValueError(f"public pretrain source {source['name']} needs local_file or local_glob")
    if source["license"] not in ALLOWED_LICENSES:
        raise ValueError(f"public pretrain source {source['name']} has disallowed license {source['license']}")
    if str(source["revision"]).strip().lower() in UNPINNED_REVISIONS:
        raise ValueError(f"public pretrain source {source['name']} must pin an immutable revision")
    if source["language"] != "en":
        raise ValueError(f"public pretrain source {source['name']} must be English")
    if source["field_policy"] != "text-only" or source["text_field"] != "text":
        raise ValueError(f"public pretrain source {source['name']} must be text-only")
    if "prompt" not in source.get("excluded_fields", []) or "seed_data" not in source.get("excluded_fields", []):
        raise ValueError(f"public pretrain source {source['name']} must exclude prompt and seed_data")
    if int(source["token_budget"]) <= 0:
        raise ValueError(f"public pretrain source {source['name']} must have positive token budget")


def prepare_public_pretrain(paths, target_tokens: int | None = None) -> Path:
    validate_public_pretrain_sources(paths)
    sources = public_pretrain_sources()
    target_tokens = public_pretrain_target(sources, target_tokens)
    output = Path(os.environ.get("PUBLIC_PRETRAIN_OUTPUT_DIR", str(paths.public_pretrain)))
    root_value = os.environ.get("TRAIN_PUBLIC_DATA_DIR", "")
    root = Path(root_value) if root_value else None
    state = CorpusState(output, target_tokens)
    for source in sources:
        for local in source_paths(root, source):
            for text in iter_source_text(local, source["text_field"]):
                if state.done:
                    break
                state.add_row(source, text)
            if state.done:
                break
    state.close()
    write_report(output, state)
    if state.tokens < target_tokens:
        raise RuntimeError(f"public pretrain corpus has {state.tokens} tokens; target is {target_tokens}")
    return output


def public_pretrain_target(sources: list[dict], target_tokens: int | None) -> int:
    if target_tokens is not None:
        return target_tokens
    default = sum(int(source["token_budget"]) for source in sources)
    return int(os.environ.get("TRAIN_PUBLIC_PRETRAIN_TOKENS", str(default)))


def pretrain_row(source: dict, text: str, index: int) -> dict:
    row_id = f"public-pretrain-{source['name']}-{index:09d}"
    return {
        "id": row_id, "mode": "pretrain", "language": "en", "domain": source["name"],
        "difficulty": "mixed", "title": title_from(text, row_id), "text": text,
        "metadata": pretrain_metadata(source, text),
    }


def pretrain_metadata(source: dict, text: str) -> dict:
    return {
        "source": "public_pretrain", "mode": "pretrain", "provenance": "public-pretrain",
        "author_type": "external", "author_model": "unknown", "language": "en",
        "license": source["license"], "source_ref": f"{source['source_url']}@{source['revision']}",
        "dataset": source["dataset"], "config": source["config"], "field_policy": source["field_policy"],
        "excluded_fields": source.get("excluded_fields", []), "estimated_tokens": approx_tokens(text),
    }


def title_from(text: str, fallback: str) -> str:
    first = text.strip().splitlines()[0].strip("# ").strip()
    return first[:80] if first else fallback


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class CorpusState:
    def __init__(self, output: Path, target_tokens: int):
        self.output, self.target_tokens = output, target_tokens
        self.rows = self.tokens = self.duplicates = 0
        self.split_rows, self.sources, self.source_tokens, self.seen = Counter(), Counter(), Counter(), set()
        self.handles = {}
        for split in ["train", "val", "holdout"]:
            (output / split).mkdir(parents=True, exist_ok=True)
            for old in (output / split).glob("*.jsonl"):
                old.unlink()

    @property
    def done(self) -> bool:
        return self.tokens >= self.target_tokens

    def add_row(self, source: dict, text: str) -> None:
        row_tokens = approx_tokens(text)
        if self.source_tokens[source["name"]] >= int(source["token_budget"]):
            return
        digest = hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()
        if digest in self.seen:
            self.duplicates += 1
            return
        self.seen.add(digest)
        row = pretrain_row(source, text, self.rows + 1)
        split = split_for(self.rows + 1)
        self.write(split, row)
        self.rows += 1
        self.split_rows[split] += 1
        self.sources[(row["domain"], row["metadata"]["license"])] += 1
        self.source_tokens[source["name"]] += row_tokens
        self.tokens += approx_tokens(row_text(row))

    def write(self, split: str, row: dict) -> None:
        chunk = self.split_rows[split] // 1000 + 1
        path = self.output / split / f"{split}-{chunk:06d}.jsonl"
        if path not in self.handles:
            self.handles[path] = path.open("a", encoding="utf-8")
        self.handles[path].write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def split_for(index: int) -> str:
    if index % 100 == 0:
        return "holdout"
    if index % 50 == 0:
        return "val"
    return "train"


def normalized_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def write_report(output: Path, state: CorpusState) -> None:
    duplicate_rate = state.duplicates / max(1, state.rows + state.duplicates)
    manifest = {
        "schema": "lkjai-public-pretrain-v1",
        "target_tokens": state.target_tokens,
        "approx_tokens": state.tokens,
        "rows": state.rows,
        "duplicate_rows": state.duplicates,
        "duplicate_rate": duplicate_rate,
        "split_rows": dict(state.split_rows),
        "source_tokens": dict(state.source_tokens),
        "field_policy": "text-only",
        "excluded_fields": ["prompt", "seed_data"],
        "sources": [{"name": name, "license": license, "rows": rows} for (name, license), rows in sorted(state.sources.items())],
        "checksums": checksums(output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "validation-report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def checksums(output: Path) -> dict:
    values = {}
    for path in sorted(output.rglob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        values[str(path.relative_to(output))] = digest
    return values
