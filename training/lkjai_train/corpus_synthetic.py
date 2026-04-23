import itertools

from .corpus_shared import split_for, xml_prompt
from .public_data import ANGLES, AUDIENCES, CONSTRAINTS, DELIVERABLES, GENERAL_TOPICS, SAFER_ALTERNATIVES, SAFETY_BOUNDARIES, SAFETY_REQUESTS
from .rows import direct_row, meta, tool_row


TOOL_SCENARIOS = [
    ("List the docs directory.", "fs.list", {"path": "docs"}, "docs\nREADME.md", "The workspace docs directory and README are available."),
    ("List the project root.", "fs.list", {"path": "."}, "README.md\nsrc\ndocs", "The workspace root contains README.md, src, and docs."),
    ("Read the main README.", "fs.read", {"path": "README.md"}, "# lkjai\n\nDocs-first scratch agent.", "The README describes a docs-first scratch agent."),
    ("Read the docs README.", "fs.read", {"path": "docs/README.md"}, "# Docs\n\nCanonical project contracts.", "The docs README acts as the project table of contents."),
    ("Store a preference about concise answers.", "memory.write", {"content": "User prefers concise answers."}, "User prefers concise answers.", "I recorded the concise-answer preference."),
    ("Search memory for preferences.", "memory.search", {"query": "preferences"}, "User prefers concise answers.", "The stored preference says answers should stay concise."),
    ("Fetch a public page.", "web.fetch", {"url": "https://example.com"}, "<html>Example Domain</html>", "The fetched page is Example Domain."),
    ("Run pwd in the workspace.", "shell.exec", {"command": "pwd"}, "/app/data/workspace", "The shell runs inside the workspace directory."),
]

TOOL_VARIANTS = [
    ("Return the correct tool, keep paths workspace-relative.", "tool_trajectory"),
    ("Use the tool before answering.", "tool_trajectory"),
    ("Prefer the typed tool call instead of prose.", "tool_selection"),
    ("Keep the action JSON minimal and valid.", "action_json"),
]


def general_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(GENERAL_TOPICS, AUDIENCES, DELIVERABLES, CONSTRAINTS, ANGLES)
    for index, ((topic, practice, failure), audience, deliverable, constraint, angle) in enumerate(combos, start=1):
        row_id = f"general-{index:05d}"
        prompt = xml_prompt(f"Explain {topic} for {audience} as {deliverable}.", f"<topic>{topic}</topic><audience>{audience}</audience><angle>{angle}</angle>", f"{constraint}; {angle}.")
        answer = f"For {audience}, {topic} should center on {practice}. Keep attention on {failure}, {angle}, and end with one concrete verification step."
        tags = ["direct_answer", "general_reasoning", "language:en", deliverable.replace(" ", "_"), audience.split()[1]]
        rows.append(direct_row(prompt, answer, tags, meta(row_id, "general-reasoning", deliverable.replace(" ", "-"), "synthetic/general", split=split_for(row_id), license_name="Apache-2.0")))
        if len(rows) >= limit:
            return rows
    return rows


def safety_rows(limit: int, rules: list[str]) -> list[dict]:
    rows = []
    combos = itertools.product(SAFETY_BOUNDARIES, SAFETY_REQUESTS, rules, SAFER_ALTERNATIVES)
    for index, (boundary, request, rule, safer) in enumerate(combos, start=1):
        row_id = f"safety-{index:05d}"
        prompt = xml_prompt(request.format(boundary=boundary), f"<boundary>{boundary}</boundary><rule>{rule}</rule>", "State the restriction and the safer alternative.")
        answer = f"Do not treat {boundary} as generally writable or publishable state. Preserve the rule that {rule}, then {safer}."
        rows.append(direct_row(prompt, answer, ["direct_answer", "language:en", "safety", "visibility_boundary"], meta(row_id, "safety-policy", "boundary-answer", "synthetic/safety", split=split_for(row_id), safety_scope="restricted")))
        if len(rows) >= limit:
            return rows
    return rows


def local_tool_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(TOOL_SCENARIOS, TOOL_VARIANTS, ANGLES, AUDIENCES)
    for index, ((prompt_text, tool, args, result, answer), (constraint, tag), angle, audience) in enumerate(combos, start=1):
        row_id = f"tool-{index:05d}"
        prompt = xml_prompt(prompt_text, f"<tool>{tool}</tool><audience>{audience}</audience><angle>{angle}</angle>", f"{constraint} {angle}.")
        tags = ["language:en", tag, tool, "workspace_tool" if tool.startswith("fs.") else "runtime_tool"]
        rows.append(tool_row(prompt, tool, args, result, answer, tags, meta(row_id, "runtime-tools", tool.replace(".", "-"), "synthetic/tools", split=split_for(row_id), toolset="local")))
        if len(rows) >= limit:
            return rows
    return rows
