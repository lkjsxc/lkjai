import json
from pathlib import Path

from .corpus import generate_corpus, source_metadata, split_rows
from .rows import direct_row, meta, signature, tool_only_row


FIXTURES = [
    direct_row(
        "<task><request>Say hello.</request><context><mode>smoke</mode></context><constraints>Return one JSON action.</constraints></task>",
        "Hello.",
        ["direct_answer", "fixture", "language:en"],
        meta("fixture-0001", "fixtures", "direct-answer", "training/tests", split="train"),
    ),
    direct_row(
        "<task><request>Explain the write policy.</request><context><mode>smoke</mode></context><constraints>Mention confirmation.</constraints></task>",
        "Writes require an explicit confirmation step before the mutation runs.",
        ["direct_answer", "fixture", "language:en"],
        meta("fixture-0002", "fixtures", "policy-answer", "training/tests", split="val"),
    ),
    tool_only_row(
        "<task><request>Search kjxlkj resources.</request><context><query>research</query></context><constraints>Return the correct tool.</constraints></task>",
        "resource.search",
        {"query": "research", "kind": "all"},
        ["fixture", "kjxlkj", "resource.search"],
        meta("fixture-0003", "fixtures", "search", "training/tests", split="holdout", toolset="kjxlkj"),
    ),
]

REQUIRED_META = {
    "id",
    "split",
    "provenance",
    "author_type",
    "author_model",
    "quality_tier",
    "domain",
    "skill",
    "toolset",
    "language",
    "safety_scope",
    "license",
    "source_ref",
}


def prepare_fixtures(paths) -> Path:
    paths.ensure()
    write_rows(paths.fixtures, FIXTURES)
    write_rows(paths.train_dataset, [FIXTURES[0]])
    write_rows(paths.val_dataset, [FIXTURES[1]])
    write_rows(paths.holdout_dataset, [FIXTURES[2]])
    write_metadata(paths, FIXTURES)
    return paths.fixtures


def prepare_corpus(paths, size: int = 12000, seed: int = 42) -> Path:
    paths.ensure()
    rows = generate_corpus(size, seed)
    split = split_rows(rows)
    write_rows(paths.corpus, rows)
    write_rows(paths.train_dataset, split["train"])
    write_rows(paths.val_dataset, split["val"])
    write_rows(paths.holdout_dataset, split["holdout"])
    write_metadata(paths, rows)
    return paths.corpus


def validate_dataset(path: Path) -> Path:
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("row missing messages")
        if not isinstance(row.get("tags"), list):
            raise ValueError("row missing tags")
        metadata = row.get("meta")
        if not isinstance(metadata, dict) or REQUIRED_META - metadata.keys():
            raise ValueError("row missing required meta fields")
        if metadata.get("split") not in {"train", "val", "holdout"}:
            raise ValueError("invalid split")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("invalid role")
            if not isinstance(message.get("content"), str):
                raise ValueError("message content must be string")
        rows += 1
    if rows == 0:
        raise ValueError("dataset is empty")
    return path


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(paths, rows: list[dict]) -> None:
    split_counts = {"train": 0, "val": 0, "holdout": 0}
    unique_rows = {signature(row) for row in rows}
    for row in rows:
        split_counts[row["meta"]["split"]] += 1
    metadata = {
        "schema": "lkjai-agent-jsonl-v2",
        "rows": len(rows),
        "split_rows": split_counts,
        "unique_rows": len(unique_rows),
        "duplicate_rows": len(rows) - len(unique_rows),
        "sources": source_metadata(rows),
    }
    paths.dataset_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
