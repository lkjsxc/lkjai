import itertools

from .corpus_shared import split_for, xml_prompt
from .rows import action_message, direct_row, kimi_meta, multi_turn_row


FAILURES = [
    ("fs.read", {"path": "docs/missing.md"}, "not found", "fs.read", {"path": "docs/README.md"}, "# Documentation Canon", "The fallback docs README contains the project canon."),
    ("fs.list", {"path": "missing"}, "No such file or directory", "fs.list", {"path": "workspace"}, "README.md", "Listed the workspace directory instead."),
    ("resource.fetch", {"ref": "missing"}, "404 Not Found", "resource.search", {"query": "missing", "kind": "all"}, "found notes", "Searched for related resources after the fetch failed."),
    ("shell.exec", {"command": "docker compose up"}, "docker daemon is not running", "shell.exec", {"command": "python3 -m pytest training/tests -m not slow"}, "passed", "Ran pytest instead because docker was unavailable."),
]

BLOCKERS = [
    ("docker unavailable", "Docker daemon is not running.", "Run dependency-light checks instead of docker compose verify."),
    ("gpu unavailable", "CUDA is not accessible.", "Set TRAIN_PRESET=quick for CPU-only smoke."),
    ("missing dependency", "torch or tokenizers is not installed.", "Skip training-dependent tests and validate sources only."),
]


def kimi_recovery_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(FAILURES, range(2))
    for index, ((first_tool, first_args, first_result, second_tool, second_args, second_result, final), variant) in enumerate(combos, start=1):
        row_id = f"kimi-recovery-{index:06d}"
        task = f"Recover from {first_tool} failure case {index}."
        prompt = xml_prompt(task, f"<first>{first_tool}</first><error>{first_result}</error>", "Revise after failure.")
        messages = [
            {"role": "user", "content": prompt},
        ]
        if variant == 1:
            messages.append(action_message({"kind": "plan", "content": f"Run {first_tool}, observe error, then fallback to {second_tool}."}))
        messages.extend([
            action_message({"kind": "tool_call", "thought": f"try {first_tool}", "tool": first_tool, "args": first_args}),
            {"role": "tool", "name": first_tool, "content": first_result},
            action_message({"kind": "tool_call", "thought": f"fallback to {second_tool}", "tool": second_tool, "args": second_args}),
            {"role": "tool", "name": second_tool, "content": second_result},
            action_message({"kind": "final", "content": final}),
        ])
        rows.append(multi_turn_row(messages, ["kimi_generated", "failure_recovery", "revision", "multi_turn", "language:en"], kimi_meta(row_id, "agentic", "failure-recovery", "training/tests", split=split_for(row_id))))
        if len(rows) >= limit:
            break
    for index in range(limit - len(rows)):
        blocker, observation, fallback = BLOCKERS[index % len(BLOCKERS)]
        row_id = f"kimi-blocker-{index + 1:06d}"
        prompt = xml_prompt(f"Handle environment blocker: {blocker}.", f"<blocker>{blocker}</blocker>", "Explain the blocker and the fallback.")
        rows.append(direct_row(prompt, f"Blocker: {blocker}. Observation: {observation}. Fallback: {fallback}.", ["kimi_generated", "environment_blocker", "language:en"], kimi_meta(row_id, "agentic", "env-blocker", "training/tests", split=split_for(row_id))))
    return rows[:limit]
