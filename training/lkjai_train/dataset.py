import json
from pathlib import Path

from .corpus import source_metadata, write_corpus

FIXTURES = [
    {
        "messages": [
            {"role": "user", "content": "List this directory."},
            {
                "role": "assistant",
                "content": '{"kind":"tool_call","thought":"inspect","tool":"fs.list","args":{"path":"."}}',
            },
            {"role": "tool", "name": "fs.list", "content": "README.md\nsrc"},
            {"role": "assistant", "content": '{"kind":"final","thought":"done","content":"README.md and src"}'},
        ],
        "tags": ["tool_trajectory"],
    },
    {
        "messages": [
            {"role": "user", "content": "Remember that I prefer concise plans."},
            {
                "role": "assistant",
                "content": '{"kind":"tool_call","thought":"store","tool":"memory.write","args":{"content":"User prefers concise plans."}}',
            },
        ],
        "tags": ["memory"],
    },
]


def prepare_fixtures(paths) -> Path:
    paths.ensure()
    with paths.fixtures.open("w", encoding="utf-8") as file:
        for row in FIXTURES:
            file.write(json.dumps(row) + "\n")
    metadata = {
        "schema": "lkjai-agent-jsonl-v1",
        "rows": len(FIXTURES),
        "target_rows": len(FIXTURES),
        "sources": [{"name": "fixtures", "license": "project-local", "rows": len(FIXTURES)}],
    }
    paths.dataset_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths.fixtures


def prepare_corpus(paths, size: int = 4000, seed: int = 42) -> Path:
    paths.ensure()
    path = paths.datasets / "corpus.jsonl"
    write_corpus(path, size, seed)
    rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
    metadata = {
        "schema": "lkjai-agent-jsonl-v1",
        "rows": rows,
        "target_rows": size,
        "sources": source_metadata(size, rows),
    }
    paths.dataset_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def validate_dataset(path: Path) -> Path:
    rows = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("row missing messages")
            for message in messages:
                if message.get("role") not in {"system", "user", "assistant", "tool"}:
                    raise ValueError("invalid role")
                if not isinstance(message.get("content"), str):
                    raise ValueError("message content must be string")
            rows += 1
    if rows == 0:
        raise ValueError("dataset is empty")
    return path
