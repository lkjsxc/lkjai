from .corpus_shared import split_for, xml_prompt
from .rows import action_message, confirm_row, kimi_meta, multi_turn_row


MUTATIONS = [
    ("resource.create_note", {"body": "# Draft\n\n- Item 1", "is_private": False}, "Create a public note from the supplied draft."),
    ("resource.update_resource", {"ref": "release-notes", "body": "# Updated", "is_private": False}, "Update the release-notes resource."),
    ("fs.write", {"path": "workspace/status.md", "content": "# Status\n\nVerified."}, "Write the status file."),
]


def kimi_confirm_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        tool, args, summary = MUTATIONS[index % len(MUTATIONS)]
        row_id = f"kimi-confirm-{index + 1:06d}"
        task = f"Perform mutation: {summary}"
        prompt = xml_prompt(task, f"<tool>{tool}</tool>", "Request confirmation before mutation.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "request_confirmation", "thought": "mutation needs confirmation", "summary": summary, "operation": tool, "pending_tool_call": {"tool": tool, "args": args}}),
            {"role": "user", "content": "<task><request>Confirmed.</request></task>"},
            action_message({"kind": "tool_call", "thought": "execute confirmed mutation", "tool": tool, "args": args}),
            {"role": "tool", "name": tool, "content": "success"},
            action_message({"kind": "final", "content": f"Mutation completed: {summary}"}),
        ]
        rows.append(multi_turn_row(messages, ["kimi_generated", "confirmation", "safe_mutation", "multi_turn", "language:en"], kimi_meta(row_id, "kjxlkj-api", "confirmed-mutation", "docs/architecture/agent/schema.md", split=split_for(row_id), toolset="kjxlkj")))
    return rows
