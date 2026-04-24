from .corpus_shared import split_for, xml_prompt
from .rows import action_message, meta, row


PAIRS = [
    (
        "good-plan-references-tools",
        {"kind": "plan", "content": "Read docs, then run tests."},
        {"kind": "plan", "content": "Do the task."},
    ),
    (
        "good-final-grounded",
        {"kind": "final", "content": "The default TRAIN_CORPUS_SIZE is 60000 per corpus.md."},
        {"kind": "final", "content": "It depends on the configuration."},
    ),
    (
        "good-tool-call-correct",
        {"kind": "tool_call", "thought": "read file", "tool": "fs.read", "args": {"path": "docs/README.md"}},
        {"kind": "tool_call", "thought": "read file", "tool": "fs.list", "args": {"path": "docs"}},
    ),
    (
        "good-confirmation-explicit",
        {"kind": "request_confirmation", "summary": "Create note", "operation": "resource.create_note", "pending_tool_call": {"tool": "resource.create_note", "args": {"body": "# Draft"}}},
        {"kind": "final", "content": "Created the note."},
    ),
    (
        "good-revision-fallback",
        {"kind": "tool_call", "thought": "fallback", "tool": "fs.read", "args": {"path": "docs/README.md"}},
        {"kind": "tool_call", "thought": "retry same path", "tool": "fs.read", "args": {"path": "docs/missing.md"}},
    ),
    (
        "good-final-cites-observation",
        {"kind": "final", "content": "The docs directory contains README.md and canon subtrees."},
        {"kind": "final", "content": "Here is the information you requested."},
    ),
    (
        "good-plan-has-stop-rule",
        {"kind": "plan", "content": "Read corpus.md, verify the stop rule, then report."},
        {"kind": "plan", "content": "Read docs and report."},
    ),
]


def preference_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        skill, chosen, rejected = PAIRS[index % len(PAIRS)]
        row_id = f"pref-{index + 1:05d}"
        prompt = xml_prompt(f"Choose the better action for case {index + 1}.", f"<skill>{skill}</skill>", "Return the better JSON action.")
        messages = [
            {"role": "user", "content": prompt},
            action_message(chosen),
        ]
        rows.append(row(messages, ["preference", "direct_answer", skill], meta(row_id, "preferences", skill, "training/tests", split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows
