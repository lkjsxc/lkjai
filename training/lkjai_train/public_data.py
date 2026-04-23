from .rows import direct_row

PUBLIC_SOURCE_METADATA = [
    {
        "name": "lkjai-generated-public-style",
        "license": "Apache-2.0",
        "source_url": "local://generated",
        "revision": "v1",
    },
    {
        "name": "OpenAssistant OASST1 candidate",
        "license": "Apache-2.0",
        "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "revision": "candidate-not-downloaded",
    },
    {
        "name": "ToolBench candidate",
        "license": "Apache-2.0",
        "source_url": "https://github.com/OpenBMB/ToolBench",
        "revision": "candidate-not-downloaded",
    },
]

PUBLIC_PROMPTS = [
    ("Explain containers briefly.", "Containers package an app with its runtime dependencies."),
    ("Give a concise checklist for debugging.", "Reproduce, inspect logs, isolate the change, fix, and rerun tests."),
    ("What makes an answer useful?", "It should be correct, direct, scoped to the request, and clear about uncertainty."),
    ("Summarize a meeting note.", "Identify decisions, action items, owners, and unresolved questions."),
]


def public_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        user, answer = PUBLIC_PROMPTS[index % len(PUBLIC_PROMPTS)]
        rows.append(direct_row(user, answer, ["public_instruction", "license:apache-2.0"]))
    return rows
