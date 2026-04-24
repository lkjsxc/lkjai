from .corpus_shared import split_for, xml_prompt
from .rows import confirm_row, direct_row, meta, revise_row, tool_row


TOOLS = [
    ("fs.list", {"path": "docs"}, "README.md\narchitecture\noperations", "The docs directory contains README.md and canon subtrees."),
    ("fs.read", {"path": "docs/README.md"}, "# Documentation Canon\n\n`docs/` is the only active canon.", "The docs README is the active project canon."),
    ("memory.write", {"content": "User prefers concise plans."}, "User prefers concise plans.", "I recorded the concise-plan preference."),
    ("memory.search", {"query": "concise plans"}, "User prefers concise plans.", "The stored preference says plans should stay concise."),
]

SCHEMAS = [
    ("final", "Return a final answer.", "Done."),
    ("tool_call", "Read a workspace file.", "fs.read"),
    ("plan", "Plan before using tools.", "Read docs, then summarize the relevant contract."),
]


def repo_schema_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        kind, request, answer = SCHEMAS[index % len(SCHEMAS)]
        row_id = f"schema-{index + 1:05d}"
        prompt = xml_prompt(request, f"<schema>{kind}</schema><case>{row_id}</case>", "Return one valid JSON action.")
        metadata = meta(row_id, "runtime-schema", kind, "src/agent/schema.rs", split=split_for(row_id))
        if kind == "tool_call":
            rows.append(tool_row(prompt, "fs.read", {"path": "README.md"}, "ok", "The file read completed.", ["runtime_schema", "tool_trajectory", "language:en"], metadata))
        else:
            rows.append(direct_row(prompt, answer, ["runtime_schema", "direct_answer", "language:en"], metadata))
    return rows


def fixture_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"fixture-extra-{index + 1:05d}"
        tool, args, result, answer = TOOLS[index % len(TOOLS)]
        prompt = xml_prompt(f"Use {tool} for repository task {index + 1}.", f"<tool>{tool}</tool><case>{row_id}</case>", "Use the tool before answering.")
        rows.append(tool_row(prompt, tool, args, result, answer, ["fixture", "tool_trajectory", "language:en"], meta(row_id, "fixtures", tool.replace(".", "-"), "training/tests", split=split_for(row_id), toolset="local")))
    rows += confirmation_rows(max(1, limit // 10))
    rows += revision_rows(max(1, limit // 10))
    return rows[:limit]


def confirmation_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"confirm-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Create a note from draft {index + 1}.", f"<draft># Status\n\n- Verified docs.</draft><case>{row_id}</case>", "Request confirmation before mutation.")
        rows.append(confirm_row(prompt, "resource.create_note", "resource.create_note", {"body": "# Status\n\n- Verified docs.", "is_private": False}, "Create the note after explicit confirmation.", ["fixture", "confirmation", "kjxlkj"], meta(row_id, "fixtures", "confirmation", "training/tests", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def revision_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"revision-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Read missing policy file case {index + 1}, then recover.", f"<path>docs/policy.md</path><case>{row_id}</case>", "Revise after a failed read.")
        rows.append(revise_row(prompt, "Try the requested path, then fall back to docs README.", "fs.read", {"path": "docs/policy.md"}, "not found", "fs.read", {"path": "docs/README.md"}, "# Documentation Canon", "The fallback docs README contains the project canon.", ["fixture", "revision", "multi_turn"], meta(row_id, "fixtures", "revision", "training/tests", split=split_for(row_id), toolset="local")))
    return rows
