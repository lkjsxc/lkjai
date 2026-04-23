import json
import os
import random
from pathlib import Path

from .docs_data import docs_rows
from .kjxlkj_data import kjxlkj_rows
from .public_data import PUBLIC_SOURCE_METADATA, public_rows
from .rows import direct_row, tool_row

_TOOL_TRAJECTORIES = [
    (
        "List the files in the current directory.",
        "fs.list",
        {"path": "."},
        "README.md\nCargo.toml\nsrc",
        "The current directory contains README.md, Cargo.toml, and a src folder.",
    ),
    (
        "Read the README file.",
        "fs.read",
        {"path": "README.md"},
        "# lkjai\n\nLocal agentic AI system.",
        "The README describes lkjai as a local agentic AI system.",
    ),
    (
        "Create a file named notes.txt with buy milk.",
        "fs.write",
        {"path": "notes.txt", "content": "buy milk"},
        "wrote notes.txt",
        "I created notes.txt with the requested content.",
    ),
    (
        "Run pwd.",
        "shell.exec",
        {"command": "pwd"},
        "/app/data/workspace",
        "The command ran inside /app/data/workspace.",
    ),
    (
        "Fetch https://example.com.",
        "web.fetch",
        {"url": "https://example.com"},
        "<html><body>Example Domain</body></html>",
        "The page at example.com contains Example Domain.",
    ),
    (
        "Remember that I prefer concise answers.",
        "memory.write",
        {"content": "User prefers concise answers."},
        "User prefers concise answers.",
        "Noted. I will keep my answers concise.",
    ),
    (
        "Remember that my favorite editor is vim.",
        "memory.write",
        {"content": "User favorite editor is vim."},
        "User favorite editor is vim.",
        "Noted. I will remember that your favorite editor is vim.",
    ),
    (
        "What do you remember about my preferences?",
        "memory.search",
        {"query": "preferences"},
        "User prefers concise answers.",
        "I remember that you prefer concise answers.",
    ),
]

_DIRECT_ANSWERS = [
    ("What is 2+2?", "2+2 equals 4."),
    ("Say hello.", "Hello!"),
    ("What is lkjai?", "lkjai is a docs-first scratch-model agent research system."),
    (
        "Please summarize the project training pipeline.",
        "The pipeline prepares corpus data, trains a tokenizer, trains the scratch model, exports artifacts, and runs fixed plus behavioral evals.",
    ),
    (
        "Why use JSON actions?",
        "JSON actions give the runtime typed fields it can validate before executing tools.",
    ),
    (
        "Why use tagged prompt sections?",
        "Tagged sections separate summaries, memories, run metadata, and recent events for the model.",
    ),
]


def generate_corpus(size: int = 4000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    targets = corpus_targets(size)
    rows = []
    rows.extend(repeat_docs(targets["docs"]))
    rows.extend(repeat_rows([direct_row(*item) for item in _DIRECT_ANSWERS], targets["direct"], rng))
    tool_rows = [tool_row(*item) for item in _TOOL_TRAJECTORIES]
    rows.extend(repeat_rows(tool_rows, targets["tools"], rng))
    rows.extend(kjxlkj_rows(targets["kjxlkj"]))
    rows.extend(public_rows(targets["public"]))
    rng.shuffle(rows)
    return rows[:size]


def corpus_targets(size: int) -> dict[str, int]:
    docs = int(size * 0.15)
    tools = int(size * 0.35)
    kjxlkj = int(size * 0.10)
    public = int(size * 0.10)
    direct = max(0, size - docs - tools - kjxlkj - public)
    return {"docs": docs, "tools": tools, "kjxlkj": kjxlkj, "public": public, "direct": direct}


def repeat_docs(limit: int) -> list[dict]:
    root = Path(os.environ.get("REPO_DOCS_DIR", "/workspace/docs"))
    base = docs_rows(root, 5000) or [direct_row("What is lkjai?", _DIRECT_ANSWERS[2][1])]
    return repeat_rows(base, limit, random.Random(7))


def repeat_rows(base: list[dict], limit: int, rng: random.Random) -> list[dict]:
    if not base or limit <= 0:
        return []
    return [base[rng.randrange(len(base))] for _ in range(limit)]


def source_metadata(size: int, rows: int) -> list[dict]:
    targets = corpus_targets(size)
    return [
        {"name": "lkjai-docs", "license": "project-local", "rows": targets["docs"]},
        {"name": "lkjai-tool-trajectories", "license": "project-local", "rows": targets["tools"]},
        {"name": "kjxlkj-doc-trajectories", "license": "project-local", "rows": targets["kjxlkj"]},
        {"name": "lkjai-direct-answers", "license": "project-local", "rows": targets["direct"]},
        {**PUBLIC_SOURCE_METADATA[0], "rows": targets["public"]},
        *PUBLIC_SOURCE_METADATA[1:],
        {"name": "actual-written-rows", "license": "n/a", "rows": rows},
    ]


def write_corpus(path: Path, size: int = 4000, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_corpus(size, seed)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
