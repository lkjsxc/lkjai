import itertools

from .corpus_shared import split_for, xml_prompt
from .rows import action_message, kimi_meta, multi_turn_row


TOOLS = [
    ("fs.read", {"path": "docs/README.md"}, "# Documentation Canon\n\n`docs/` is the only active canon."),
    ("fs.list", {"path": "src/agent"}, "schema.rs\naction.rs\ntools.rs"),
    ("memory.search", {"query": "preference"}, "User prefers concise plans."),
    ("resource.search", {"query": "research", "kind": "all"}, "release-notes\nresearch-paper"),
    ("resource.fetch", {"ref": "release-notes"}, "# Release Notes\n\n- Added XML action schema."),
    ("shell.exec", {"command": "cargo test", "timeout": 120}, "test result: ok. 14 passed"),
]

SCENARIOS = [
    ("read-file", "Read the docs README.", "fs.read", {"path": "docs/README.md"}),
    ("list-dir", "List the agent source directory.", "fs.list", {"path": "src/agent"}),
    ("search-memory", "Search for user preferences.", "memory.search", {"query": "preference"}),
    ("search-resource", "Search kjxlkj resources for research.", "resource.search", {"query": "research", "kind": "all"}),
    ("fetch-resource", "Fetch the release-notes resource.", "resource.fetch", {"ref": "release-notes"}),
    ("run-tests", "Run the Rust test suite.", "shell.exec", {"command": "cargo test", "timeout": 120}),
]


def kimi_schema_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(SCENARIOS, range(2))
    for index, ((skill, task, expected_tool, expected_args), variant) in enumerate(combos, start=1):
        row_id = f"kimi-schema-{index:06d}"
        prompt = xml_prompt(task, f"<scenario>{skill}</scenario>", "Return the canonical XML tool call.")
        tool, args, obs = TOOLS[index % len(TOOLS)]
        messages = [
            {"role": "user", "content": prompt},
        ]
        if variant == 1:
            messages.append(action_message({"kind": "plan", "content": f"Select {expected_tool} with correct arguments."}))
        messages.append(action_message({"kind": "tool_call", "thought": f"use {expected_tool}", "tool": expected_tool, "args": expected_args}))
        messages.append({"role": "tool", "name": expected_tool, "content": obs})
        messages.append(action_message({"kind": "final", "content": f"Executed {expected_tool} successfully. Observation: {obs[:120]}"}))
        rows.append(multi_turn_row(messages, ["kimi_generated", "runtime_schema", "tool_selection", "language:en", skill], kimi_meta(row_id, "runtime-schema", skill, "docs/architecture/agent/schema.md", split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows
