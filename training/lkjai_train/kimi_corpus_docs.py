import itertools
from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .docs_data import doc_records
from .rows import action_message, kimi_meta, multi_turn_row


DOC_TASKS = [
    ("summarize", "Summarize the contract in {path}.", "Ground the answer in the file content."),
    ("default", "What is the default behavior in {path}?", "State exact defaults."),
    ("verify", "What verification applies to {path}?", "Name the concrete command."),
    ("risk", "What failure mode does {path} prevent?", "Connect to runtime behavior."),
    ("audience", "Who is the intended audience of {path}?", "Name the primary reader."),
    ("change", "What must be updated when {path} changes?", "List downstream artifacts."),
]

FOLLOWUP_TOOLS = [
    ("fs.read", {"path": "docs/README.md"}),
    ("fs.list", {"path": "docs/architecture"}),
    ("fs.read", {"path": "docs/architecture/agent/schema.md"}),
]


def kimi_doc_rows(limit: int) -> list[dict]:
    root = Path(__file__).resolve().parents[2] / "docs"
    records = doc_records(root)
    rows = []
    combos = itertools.product(records, DOC_TASKS, range(3))
    for index, (record, (skill, request, constraint), variant) in enumerate(combos, start=1):
        row_id = f"kimi-docs-{index:06d}"
        prompt = xml_prompt(request.format(path=record["path"]), f"<path>{record['path']}</path><title>{record['title']}</title>", constraint)
        plan = f"Read {record['path']}, then summarize the relevant contract."
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": plan}),
            action_message({"kind": "tool_call", "thought": f"read {record['path']}", "tool": "fs.read", "args": {"path": record["path"]}}),
            {"role": "tool", "name": "fs.read", "content": record["snippet"][:400]},
        ]
        if variant > 0:
            tool, args = FOLLOWUP_TOOLS[(index - 1) % len(FOLLOWUP_TOOLS)]
            messages.append(action_message({"kind": "tool_call", "thought": f"follow up with {tool}", "tool": tool, "args": args}))
            messages.append({"role": "tool", "name": tool, "content": record.get("defaults", "ok") or "ok"})
        answer = f"For {record['path']}, {skill}: {record['snippet'][:240]}"
        messages.append(action_message({"kind": "final", "content": answer}))
        rows.append(multi_turn_row(messages, ["kimi_generated", "docs_grounding", "multi_turn", "language:en", skill], kimi_meta(row_id, "lkjai-docs", skill, record["path"], split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows
