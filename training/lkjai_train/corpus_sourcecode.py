from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .rows import direct_row, meta, tool_row


SOURCE_FILES = [
    "src/agent/schema.rs", "src/agent/action.rs", "src/agent/tools.rs",
    "src/agent/confirmation.rs", "src/agent/memory.rs", "src/agent/prompt.rs",
    "src/config.rs", "src/lib.rs", "src/main.rs", "src/model_client.rs",
    "Cargo.toml", "docker-compose.yml", "README.md", "verify.sh", ".env.example",
]

TEMPLATES = [
    ("summarize-file", "Summarize the purpose of {path}.", "Ground the answer in the file content."),
    ("identify-struct", "What structs or enums are defined in {path}?", "List at least one."),
    ("find-default", "What defaults are configured in {path}?", "State exact values."),
    ("dependency", "What does {path} depend on?", "Name upstream dependencies."),
    ("tool-mapping", "What tools or routes are defined in {path}?", "Name concrete tools."),
    ("verify-command", "What verification command applies to {path}?", "Name the exact command."),
    ("risk", "What failure mode does {path} prevent?", "Connect to runtime behavior."),
    ("change-impact", "What must be updated when {path} changes?", "List downstream artifacts."),
    ("owner", "Which component owns the contract in {path}?", "Name the owner."),
    ("behavior", "What behavior does {path} implement?", "Be specific about inputs and outputs."),
]

ANGLES = [
    "state the canonical default",
    "name the likely regression",
    "identify the verification command",
    "explain the implementation consequence",
    "summarize the source of truth",
    "map the change to an existing decision",
]


def sourcecode_rows(limit: int) -> list[dict]:
    rows = []
    root = Path(__file__).resolve().parents[2]
    for index, rel in enumerate(_cycle(SOURCE_FILES, limit)):
        if len(rows) >= limit:
            break
        path = root / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("//")]
        for skill, request, constraint in TEMPLATES:
            for angle in ANGLES:
                row_id = f"src-{index:05d}-{skill}-{len(rows) + 1:05d}"
                window = (len(rows) // (len(TEMPLATES) * len(ANGLES))) % max(1, len(lines))
                snippet = " ".join(lines[window:window + 6])[:400]
                context = f"<path>{rel}</path><snippet>{snippet[:180]}</snippet><angle>{angle}</angle><case>{row_id}</case>"
                prompt = xml_prompt(request.format(path=rel), context, f"{constraint} {angle}.")
                answer = f"{rel}: {snippet[:280]}"
                rows.append(direct_row(prompt, answer, ["repo_derived", "codebase_grounding", skill], meta(row_id, "source-code", skill, rel, split=split_for(row_id))))
                if len(rows) >= limit:
                    return rows
    return rows


def _cycle(items: list, limit: int):
    idx = 0
    while idx < limit:
        yield items[idx % len(items)]
        idx += 1
