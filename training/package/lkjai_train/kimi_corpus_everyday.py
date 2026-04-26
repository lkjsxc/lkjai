from .corpus_shared import split_for
from .rows import action_message, kimi_meta, row


OPENERS = [
    ("Hello", "Hello. How can I help?"),
    ("Hi there", "Hi. What would you like to work on?"),
    ("Good morning", "Good morning. What do you need today?"),
    ("Thanks", "You're welcome."),
    ("Thank you", "You're welcome."),
    ("What can you help me with?", "I can chat, remember preferences, inspect files, and use local tools when needed."),
]

PREFERENCES = [
    ("Please keep answers concise.", "Got it. I will keep answers concise."),
    ("Use short plans.", "Understood. I will keep plans short."),
    ("I prefer direct answers.", "Noted. I will answer directly when tools are not needed."),
]

CLARIFY = [
    ("Can you explain that more simply?", "Yes. I will use simpler wording."),
    ("What do you mean?", "I mean I will choose the smallest useful next step and explain it plainly."),
    ("Can you summarize?", "Yes. I will summarize the key point."),
]

LKJAI = [
    ("Explain lkjai in one sentence.", "lkjai is a local scratch-model agent with XML actions, tools, memory, and a web chat UI."),
    ("Do you need tools for every question?", "No. For ordinary chat I should answer directly with agent.finish."),
    ("Should you show reasoning?", "Yes. I should show only a brief visible rationale, not hidden chain-of-thought."),
]


def kimi_everyday_rows(limit: int) -> list[dict]:
    base = OPENERS + PREFERENCES + CLARIFY + LKJAI
    rows = []
    for index in range(limit):
        prompt, answer = base[index % len(base)]
        row_id = f"kimi-chat-{index + 1:06d}"
        variant = index // len(base)
        user = prompt if variant == 0 else f"{prompt} (conversation case {variant})"
        final = action_message({"kind": "final", "thought": "answer directly", "content": answer})
        rows.append(
            row(
                [{"role": "user", "content": user}, final],
                ["kimi_generated", "everyday_chat", "direct_answer", "language:en"],
                kimi_meta(row_id, "everyday-chat", "basic-conversation", "docs/operations/training/agent-assessment.md", split=split_for(row_id)),
            )
        )
    return rows
