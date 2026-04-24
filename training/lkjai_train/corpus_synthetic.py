import itertools

from .corpus_shared import split_for, xml_prompt
from .corpus_source import tagged_contents
from .public_data import ANGLES, AUDIENCES, CONSTRAINTS, DELIVERABLES, GENERAL_TOPICS
from .rows import direct_row, meta, tool_row


FACTS = [(item["subject"], item["answer"]) for item in tagged_contents("general", "concept_fact")]
TOOL_SCENARIOS = [
    (item["prompt"], item["tool"], item["args"], item["result"], item["answer"])
    for item in tagged_contents("general", "local_tool_scenario")
]
TOOL_VARIANTS = [(item["constraint"], item["tag"]) for item in tagged_contents("general", "tool_variant")]


def general_rows(limit: int) -> list[dict]:
    rows = operational_rows(4500) + arithmetic_rows(2500) + concept_rows(1500)
    if len(rows) < limit:
        raise RuntimeError(f"general rows under target: {len(rows)}")
    return rows[:limit]


def operational_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(GENERAL_TOPICS, AUDIENCES, DELIVERABLES, CONSTRAINTS, ANGLES)
    for index, ((topic, practice, failure), audience, deliverable, constraint, angle) in enumerate(combos, start=1):
        row_id = f"ops-{index:05d}"
        prompt = xml_prompt(f"Explain {topic} for {audience} as {deliverable}.", f"<topic>{topic}</topic><audience>{audience}</audience><angle>{angle}</angle>", f"{constraint}; {angle}.")
        answer = f"For {audience}, {topic} should center on {practice}. Keep attention on {failure}, {angle}, and end with one concrete verification step."
        tags = ["direct_answer", "general_reasoning", "language:en", deliverable.replace(" ", "_"), "operations"]
        rows.append(direct_row(prompt, answer, tags, meta(row_id, "general-reasoning", deliverable.replace(" ", "-"), "synthetic/general", split=split_for(row_id), license_name="Apache-2.0")))
        if len(rows) >= limit:
            return rows
    return rows


def arithmetic_rows(limit: int) -> list[dict]:
    rows = []
    operations = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    for index, (a, b, (symbol, op)) in enumerate(itertools.product(range(1, 46), range(1, 28), operations), start=1):
        row_id = f"math-{index:05d}"
        prompt = xml_prompt(f"What is {a} {symbol} {b}?", f"<domain>arithmetic</domain><operation>{symbol}</operation>", "Return the exact result.")
        rows.append(direct_row(prompt, str(op(a, b)), ["arithmetic", "direct_answer", "language:en"], meta(row_id, "general-reasoning", "arithmetic", "synthetic/math", split=split_for(row_id), license_name="Apache-2.0")))
        if len(rows) >= limit:
            return rows
    return rows


def concept_rows(limit: int) -> list[dict]:
    rows = []
    prompts = [
        "Explain {subject}.",
        "Summarize {subject}.",
        "State the default for {subject}.",
        "Why does {subject} matter?",
        "What is {subject}?",
    ]
    constraints = ["Return one valid JSON action.", *CONSTRAINTS]
    combos = itertools.product(FACTS, prompts, constraints, ["empty", "tagged"], ANGLES)
    for index, ((subject, answer), prompt_text, constraint, context_kind, angle) in enumerate(combos, start=1):
        row_id = f"concept-{index:05d}"
        context = "" if context_kind == "empty" else f"<subject>{subject}</subject><angle>{angle}</angle>"
        prompt = xml_prompt(prompt_text.format(subject=subject), context, constraint)
        rows.append(direct_row(prompt, answer, ["concept", "direct_answer", "language:en"], meta(row_id, "general-reasoning", "concept-answer", "synthetic/concepts", split=split_for(row_id), license_name="Apache-2.0")))
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
