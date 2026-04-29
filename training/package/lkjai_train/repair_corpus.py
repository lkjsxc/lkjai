import json
import sys
from pathlib import Path

from .rows import confirm_row, direct_row, tool_only_row, tool_row
from .rows_meta import meta


DIRECT = [
    ("Hello", "Hello. How can I help?"),
    ("Hi", "Hi. What would you like to do?"),
    ("Thanks", "You're welcome."),
    ("What can you help me with?", "I can answer questions, inspect the workspace, remember preferences, and help plan changes."),
    ("Do you need tools for every question?", "No. I use tools only when they are needed."),
    ("Please answer briefly.", "Understood. I will keep the answer brief."),
]

TOOLS = [
    ("List files in the workspace.", "fs.list", {"path": "."}),
    ("Show the README.", "fs.read", {"path": "README.md"}),
    ("Search docs for training.", "fs.search", {"path": "docs", "query": "training"}),
    ("Remember that I prefer concise answers.", "memory.write", {"key": "preference.answer_style", "value": "concise"}),
    ("What do I prefer?", "memory.read", {"key": "preference.answer_style"}),
]

CONFIRM = [
    ("Delete the temporary note.", "delete file", "fs.delete", {"path": "tmp/note.md"}, "Delete tmp/note.md."),
    ("Overwrite README with draft text.", "write file", "fs.write", {"path": "README.md", "content": "draft"}, "Overwrite README.md."),
]

OBSERVED = [
    ("List files in the workspace.", "fs.list", {"path": "."}, "README.md\ndocs\ntraining", "The workspace contains README.md, docs, and training."),
    ("What do I prefer?", "memory.read", {"key": "preference.answer_style"}, "concise", "You prefer concise answers."),
]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/data/xml-action-repair-v1")
    rows = build_rows(180_000)
    for split in ["train", "val", "holdout"]:
        write_split(root / split, split, [row for row in rows if row["meta"]["split"] == split])
    manifest = {"schema": "lkjai-xml-action-repair-v1", "rows": len(rows), "split_rows": split_counts(rows)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(root), **manifest}))


def build_rows(total: int) -> list[dict]:
    rows = []
    for index in range(total):
        split = split_for(index)
        variant = index // (len(DIRECT) + len(TOOLS) + len(CONFIRM) + len(OBSERVED) + 1)
        case = index % (len(DIRECT) + len(TOOLS) + len(CONFIRM) + len(OBSERVED) + 1)
        row_id = f"xml-repair-{index:06d}"
        rows.append(make_row(row_id, split, case, variant))
    return rows


def make_row(row_id: str, split: str, case: int, variant: int) -> dict:
    if case < len(DIRECT):
        prompt, answer = DIRECT[case]
        return direct_row(var_prompt(prompt, variant), var_answer(answer, variant), tags("direct"), meta_for(row_id, split, "direct"))
    case -= len(DIRECT)
    if case < len(TOOLS):
        prompt, tool, args = TOOLS[case]
        return tool_only_row(var_prompt(prompt, variant), tool, var_args(args, variant), tags(tool), meta_for(row_id, split, tool), thought="Use the requested tool.")
    case -= len(TOOLS)
    if case < len(CONFIRM):
        prompt, op, tool, args, summary = CONFIRM[case]
        return confirm_row(var_prompt(prompt, variant), op, tool, var_args(args, variant), summary, tags("confirm"), meta_for(row_id, split, "confirm"))
    case -= len(CONFIRM)
    if case < len(OBSERVED):
        prompt, tool, args, result, final = OBSERVED[case]
        return tool_row(var_prompt(prompt, variant), tool, var_args(args, variant), result, var_answer(final, variant), tags(tool), meta_for(row_id, split, "observed"))
    return direct_row(task_prompt("Say hello.", variant), "Hello.", tags("task"), meta_for(row_id, split, "task"))


def var_prompt(text: str, variant: int) -> str:
    marker = f"repair session {variant}"
    forms = [
        f"{text} ({marker})",
        f"Please: {text} [{marker}]",
        f"<task><request>{text}</request><context><session>{variant}</session></context><constraints>Return one XML action.</constraints></task>",
    ]
    return forms[variant % len(forms)]


def task_prompt(text: str, variant: int) -> str:
    return f"<task><request>{text}</request><context><case>{variant}</case></context><constraints>Return exactly one XML action.</constraints></task>"


def var_answer(text: str, variant: int) -> str:
    return text if variant % 2 == 0 else f"{text}"


def var_args(args: dict, variant: int) -> dict:
    return {key: (f"{value}" if key != "path" else value) for key, value in args.items()}


def meta_for(row_id: str, split: str, skill: str) -> dict:
    return meta(row_id, "xml-action-repair", skill, "docs/operations/training/xml-action-repair.md", split=split, toolset="lkjai")


def tags(kind: str) -> list[str]:
    return ["xml_action_repair", "language:en", kind]


def split_for(index: int) -> str:
    if index % 100 == 0:
        return "holdout"
    if index % 50 == 0:
        return "val"
    return "train"


def split_counts(rows: list[dict]) -> dict:
    return {split: sum(1 for row in rows if row["meta"]["split"] == split) for split in ["train", "val", "holdout"]}


def write_split(directory: Path, split: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.jsonl"):
        old.unlink()
    for start in range(0, len(rows), 1000):
        path = directory / f"{split}-{start // 1000 + 1:06d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[start : start + 1000]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
