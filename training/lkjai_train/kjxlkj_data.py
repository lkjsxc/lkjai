from .rows import direct_row, tool_row

KJXLKJ_ROWS = [
    direct_row(
        "What is kjxlkj?",
        "kjxlkj is an LLM-operated Markdown resource system for notes and media.",
        ["kjxlkj", "direct_answer"],
    ),
    direct_row(
        "How should lkjai organize kjxlkj notes?",
        "Search resources, read relevant notes, summarize clusters, then propose focused note updates.",
        ["kjxlkj", "organization"],
    ),
    tool_row(
        "Search kjxlkj for notes about deployment.",
        "fs.read",
        {"path": "kjxlkj/search-results/deployment.json"},
        '[{"id":"abc","title":"Compose deployment","is_private":false}]',
        "I found a public note titled Compose deployment.",
    ),
    tool_row(
        "Remember that private kjxlkj notes must stay private.",
        "memory.write",
        {"content": "kjxlkj private resources must not be summarized into public notes."},
        "kjxlkj private resources must not be summarized into public notes.",
        "Noted. I will keep private kjxlkj content out of public summaries.",
    ),
    tool_row(
        "Create a kjxlkj note draft about weekly research.",
        "fs.write",
        {"path": "kjxlkj/drafts/weekly-research.md", "content": "# Weekly Research\n\n- Review lkjai evals."},
        "wrote kjxlkj/drafts/weekly-research.md",
        "I drafted a weekly research note in the workspace.",
    ),
    tool_row(
        "List kjxlkj draft notes.",
        "fs.list",
        {"path": "kjxlkj/drafts"},
        "weekly-research.md\nproject-map.md",
        "The draft notes are weekly-research.md and project-map.md.",
    ),
]


def kjxlkj_rows(limit: int) -> list[dict]:
    return [KJXLKJ_ROWS[index % len(KJXLKJ_ROWS)] for index in range(limit)]
