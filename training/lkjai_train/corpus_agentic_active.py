from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .docs_data import doc_records
from .rows import action_message, meta, multi_turn_row, revise_row


PLANS = [
    "Read the requested file, then summarize the relevant contract.",
    "List the directory, then read the README to find the canon.",
    "Search memory for preferences, then answer concisely.",
    "Read the policy file, then check the provenance rules.",
    "List source files, then read the schema to identify tools.",
    "Read the config, then state the default behavior.",
    "Search docs for the contract, then verify with the command.",
]

FIRST_TOOLS = [
    ("fs.read", {"path": "{path}"}),
    ("fs.list", {"path": "{dir}"}),
    ("memory.search", {"query": "preference"}),
    ("fs.read", {"path": "docs/architecture/training/provenance.md"}),
    ("fs.list", {"path": "src/agent"}),
    ("fs.read", {"path": "src/config.rs"}),
    ("resource.search", {"query": "contract", "kind": "all"}),
]

SECOND_TOOLS = [
    ("fs.read", {"path": "docs/README.md"}),
    ("fs.read", {"path": "{path}"}),
    ("memory.write", {"content": "User prefers concise plans."}),
    ("fs.read", {"path": "docs/architecture/training/corpus.md"}),
    ("fs.read", {"path": "src/agent/schema.rs"}),
    ("fs.read", {"path": "Cargo.toml"}),
    ("resource.fetch", {"ref": "docs"}),
]


def agentic_active_rows(limit: int) -> list[dict]:
    rows = []
    root = Path(__file__).resolve().parents[2] / "docs"
    records = doc_records(root)
    for index in range(limit):
        record = records[index % len(records)]
        plan = PLANS[index % len(PLANS)]
        first_tool, first_args_tmpl = FIRST_TOOLS[index % len(FIRST_TOOLS)]
        second_tool, second_args_tmpl = SECOND_TOOLS[index % len(SECOND_TOOLS)]
        first_args = {k: v.format(path=record["path"], dir=str(Path(record["path"]).parent)) for k, v in first_args_tmpl.items()}
        second_args = {k: v.format(path=record["path"], dir=str(Path(record["path"]).parent)) for k, v in second_args_tmpl.items()}
        row_id = f"agentic-active-{index + 1:05d}"
        prompt = xml_prompt(f"Complete repository task {index + 1}.", f"<path>{record['path']}</path><title>{record['title']}</title>", "Plan before acting.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": plan}),
            action_message({"kind": "tool_call", "thought": f"run {first_tool}", "tool": first_tool, "args": first_args}),
            {"role": "tool", "name": first_tool, "content": record["snippet"][:240]},
            action_message({"kind": "tool_call", "thought": f"run {second_tool}", "tool": second_tool, "args": second_args}),
            {"role": "tool", "name": second_tool, "content": record["defaults"] or "ok"},
            action_message({"kind": "final", "content": f"Completed task for {record['path']}."}),
        ]
        rows.append(multi_turn_row(messages, ["agentic", "multi_turn", "tool_chain", "language:en"], meta(row_id, "agentic-active", "tool-chain", record["path"], split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows
