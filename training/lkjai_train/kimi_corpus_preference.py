from .corpus_shared import split_for, xml_prompt
from .rows import action_message, kimi_meta, multi_turn_row


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
        "good-revision-fallback",
        {"kind": "tool_call", "thought": "fallback", "tool": "fs.read", "args": {"path": "docs/README.md"}},
        {"kind": "tool_call", "thought": "retry same path", "tool": "fs.read", "args": {"path": "docs/missing.md"}},
    ),
]


def kimi_preference_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        skill, chosen, rejected = PAIRS[index % len(PAIRS)]
        row_id = f"kimi-pref-{index + 1:06d}"
        prompt = xml_prompt(f"Critique and improve the action for case {index + 1}.", f"<skill>{skill}</skill>", "Return the better XML action.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": f"The rejected action uses {rejected.get('tool', 'unknown')}. The chosen action is better because it is grounded and specific."}),
            action_message(rejected),
            {"role": "tool", "name": rejected.get("tool", ""), "content": "incomplete or incorrect result"},
            action_message({"kind": "plan", "content": "Revise to the preferred action."}),
            action_message(chosen),
            action_message({"kind": "final", "content": f"Selected the preferred {skill} action."}),
        ]
        rows.append(multi_turn_row(messages, ["kimi_generated", "preference", "critique", "revision", "language:en", skill], kimi_meta(row_id, "preferences", skill, "training/tests", split=split_for(row_id))))
    return rows
