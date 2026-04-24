import itertools

from .corpus_shared import split_for, xml_prompt
from .corpus_source import tagged_contents
from .public_data import ANGLES, AUDIENCES, CONSTRAINTS
from .rows import action_message, meta, multi_turn_row, revise_row


PLAN_SCENARIOS = tagged_contents("agentic_plan", "agentic_scenario")
TOOL_CHAINS = tagged_contents("agentic_tools", "agentic_tool_chain")
REVISIONS = tagged_contents("agentic_revision", "agentic_revision")


def agentic_rows(limit: int) -> list[dict]:
    rows = plan_rows(4000) + chain_rows(4500) + revision_rows(3000)
    if len(rows) < limit:
        raise RuntimeError(f"agentic rows under target: {len(rows)}")
    return rows[:limit]


def plan_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(PLAN_SCENARIOS, ANGLES, CONSTRAINTS, AUDIENCES)
    for index, (scenario, angle, constraint, audience) in enumerate(combos, start=1):
        row_id = f"agentic-plan-{index:05d}"
        task = scenario["task"]
        plan = scenario["plan"]
        tools = scenario["tools"]
        observations = scenario["observations"]
        prompt = xml_prompt(task, f"<audience>{audience}</audience><angle>{angle}</angle>", f"{constraint}; {angle}.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": plan}),
        ]
        for tool, obs in zip(tools, observations):
            messages.append(action_message({"kind": "tool_call", "thought": f"execute {tool}", "tool": tool, "args": {}}))
            messages.append({"role": "tool", "name": tool, "content": obs})
        messages.append(action_message({"kind": "final", "content": scenario["final_answer"]}))
        tags = ["agentic", "multi_turn", "planning", "language:en"]
        rows.append(multi_turn_row(messages, tags, meta(row_id, "agentic", "planning", "synthetic/agentic", split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows


def chain_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(TOOL_CHAINS, ANGLES, CONSTRAINTS)
    for index, (chain, angle, constraint) in enumerate(combos, start=1):
        row_id = f"agentic-chain-{index:05d}"
        task = f"Use {chain['first_tool']} then {chain['second_tool']}."
        plan = f"First {chain['first_tool']}, then {chain['second_tool']}."
        prompt = xml_prompt(task, f"<angle>{angle}</angle>", f"{constraint}; {angle}.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": plan}),
            action_message({"kind": "tool_call", "thought": f"run {chain['first_tool']}", "tool": chain["first_tool"], "args": chain["first_args"]}),
            {"role": "tool", "name": chain["first_tool"], "content": chain["first_observation"]},
            action_message({"kind": "tool_call", "thought": f"run {chain['second_tool']}", "tool": chain["second_tool"], "args": chain["second_args"]}),
            {"role": "tool", "name": chain["second_tool"], "content": chain["second_observation"]},
            action_message({"kind": "final", "content": chain["final_answer"]}),
        ]
        tags = ["agentic", "multi_turn", "tool_chain", "language:en"]
        rows.append(multi_turn_row(messages, tags, meta(row_id, "agentic", "tool-chain", "synthetic/agentic", split=split_for(row_id))))
        if len(rows) >= limit:
            return rows
    return rows


def revision_rows(limit: int) -> list[dict]:
    rows = []
    combos = itertools.product(REVISIONS, ANGLES, CONSTRAINTS)
    for index, (rev, angle, constraint) in enumerate(combos, start=1):
        row_id = f"agentic-rev-{index:05d}"
        prompt = xml_prompt(rev["task"], f"<angle>{angle}</angle>", f"{constraint}; {angle}.")
        rows.append(
            revise_row(
                prompt,
                rev["initial_plan"],
                rev["failed_tool"],
                rev["failed_args"],
                rev["error_observation"],
                rev["correct_tool"],
                rev["correct_args"],
                rev.get("correct_observation", "success"),
                rev["final_answer"],
                ["agentic", "multi_turn", "revision", "language:en"],
                meta(row_id, "agentic", "revision", "synthetic/agentic", split=split_for(row_id)),
            )
        )
        if len(rows) >= limit:
            return rows
    return rows
