from .corpus_shared import split_for, xml_prompt
from .rows import confirm_row, direct_row, meta, revise_row, tool_row


TOOLS = [
    ("fs.list", {"path": "docs"}, "README.md\narchitecture\noperations", "The docs directory contains README.md and canon subtrees."),
    ("fs.read", {"path": "docs/README.md"}, "# Documentation Canon\n\n`docs/` is the only active canon.", "The docs README is the active project canon."),
    ("memory.write", {"content": "User prefers concise plans."}, "User prefers concise plans.", "I recorded the concise-plan preference."),
    ("memory.search", {"query": "concise plans"}, "User prefers concise plans.", "The stored preference says plans should stay concise."),
    ("fs.list", {"path": "src/agent"}, "schema.rs\naction.rs\ntools.rs", "The agent module contains schema, action, and tools."),
    ("fs.read", {"path": "src/config.rs"}, "pub struct Config { ... }", "Config defines runtime parameters."),
    ("fs.read", {"path": "Cargo.toml"}, "[package]\nname = \"lkjai\"", "Cargo.toml declares the package and dependencies."),
    ("fs.read", {"path": "docker-compose.yml"}, "services:\n  train:\n    build: .", "Docker Compose defines train, inference, web, and verify profiles."),
]

SCHEMAS = [
    ("final", "Return a final answer.", "Done."),
    ("tool_call", "Read a workspace file.", "fs.read"),
    ("plan", "Plan before using tools.", "Read docs, then summarize the relevant contract."),
    ("final", "Summarize the verification result.", "Verification passed."),
    ("tool_call", "List files in a directory.", "fs.list"),
    ("plan", "Plan a fix for a failing test.", "Read the test, identify the failure, edit the source, rerun the test."),
    ("final", "Confirm the mutation was applied.", "Mutation completed successfully."),
    ("tool_call", "Search workspace memory.", "memory.search"),
    ("plan", "Plan how to search docs for a contract.", "List docs, read the relevant README, extract the contract."),
    ("final", "State the default corpus size.", "The default TRAIN_CORPUS_SIZE is 60000."),
    ("tool_call", "Write a preference to memory.", "memory.write"),
    ("plan", "Plan how to verify a Docker Compose setup.", "Run docker compose --profile verify up --build --abort-on-container-exit verify."),
    ("final", "Report the behavioral eval pass rate.", "The pass rate must be >= 0.80 for competency."),
    ("tool_call", "Fetch a kjxlkj resource.", "resource.fetch"),
    ("plan", "Plan how to handle a failed tool result.", "Observe the error, revise the plan, try a fallback tool."),
]

FAILURES = [
    ("fs.read", {"path": "docs/missing.md"}, "not found", "fs.read", {"path": "docs/README.md"}, "# Documentation Canon", "The fallback docs README contains the project canon."),
    ("fs.list", {"path": "missing"}, "No such file or directory", "fs.list", {"path": "workspace"}, "README.md", "Listed the workspace directory instead."),
    ("resource.fetch", {"ref": "missing"}, "404", "resource.search", {"query": "missing", "kind": "all"}, "found notes", "Searched for related resources after the fetch failed."),
]

ENV_BLOCKERS = [
    ("docker unavailable", "Docker daemon is not running.", "Run dependency-light checks instead of docker compose verify."),
    ("gpu unavailable", "CUDA is not accessible.", "Set TRAIN_PRESET=quick for CPU-only smoke."),
    ("missing dependency", "torch or tokenizers is not installed.", "Skip training-dependent tests and validate sources only."),
]


def repo_schema_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        kind, request, answer = SCHEMAS[index % len(SCHEMAS)]
        row_id = f"schema-{index + 1:05d}"
        prompt = xml_prompt(f"{request} (case {row_id})", f"<schema>{kind}</schema><case>{row_id}</case>", "Return one valid JSON action.")
        metadata = meta(row_id, "runtime-schema", kind, "src/agent/schema.rs", split=split_for(row_id))
        if kind == "tool_call":
            rows.append(tool_row(prompt, "fs.read", {"path": "README.md"}, "ok", f"The file read completed for {row_id}.", ["runtime_schema", "tool_trajectory", "language:en"], metadata))
        else:
            rows.append(direct_row(prompt, answer, ["runtime_schema", "direct_answer", "language:en"], metadata))
    return rows


def fixture_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"fixture-extra-{index + 1:05d}"
        tool, args, result, answer = TOOLS[index % len(TOOLS)]
        prompt = xml_prompt(f"Use {tool} for repository task {index + 1}.", f"<tool>{tool}</tool><case>{row_id}</case>", "Use the tool before answering.")
        rows.append(tool_row(prompt, tool, args, result, answer, ["fixture", "tool_trajectory", "language:en"], meta(row_id, "fixtures", tool.replace(".", "-"), "training/tests", split=split_for(row_id), toolset="local")))
    rows += confirmation_rows(max(1, limit // 10))
    rows += revision_rows(max(1, limit // 10))
    rows += failure_diagnosis_rows(max(1, limit // 10))
    rows += env_blocker_rows(max(1, limit // 10))
    return rows[:limit]


def confirmation_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"confirm-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Create a note from draft {index + 1}.", f"<draft># Status\n\n- Verified docs.</draft><case>{row_id}</case>", "Request confirmation before mutation.")
        rows.append(confirm_row(prompt, "resource.create_note", "resource.create_note", {"body": "# Status\n\n- Verified docs.", "is_private": False}, "Create the note after explicit confirmation.", ["fixture", "confirmation", "kjxlkj"], meta(row_id, "fixtures", "confirmation", "training/tests", split=split_for(row_id), toolset="kjxlkj")))
    return rows


def revision_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        row_id = f"revision-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Read missing policy file case {index + 1}, then recover.", f"<path>docs/policy.md</path><case>{row_id}</case>", "Revise after a failed read.")
        rows.append(revise_row(prompt, "Try the requested path, then fall back to docs README.", "fs.read", {"path": "docs/policy.md"}, "not found", "fs.read", {"path": "docs/README.md"}, "# Documentation Canon", "The fallback docs README contains the project canon.", ["fixture", "revision", "multi_turn"], meta(row_id, "fixtures", "revision", "training/tests", split=split_for(row_id), toolset="local")))
    return rows


def failure_diagnosis_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        first_tool, first_args, first_result, second_tool, second_args, second_result, final = FAILURES[index % len(FAILURES)]
        row_id = f"failure-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Diagnose failure case {index + 1}.", f"<first>{first_tool}</first><case>{row_id}</case>", "Recover from the failed tool result.")
        rows.append(revise_row(prompt, f"Run {first_tool}, then recover.", first_tool, first_args, first_result, second_tool, second_args, second_result, final, ["fixture", "revision", "failure_diagnosis"], meta(row_id, "fixtures", "failure-diagnosis", "training/tests", split=split_for(row_id), toolset="local")))
    return rows


def env_blocker_rows(limit: int) -> list[dict]:
    rows = []
    for index in range(limit):
        blocker, observation, fallback = ENV_BLOCKERS[index % len(ENV_BLOCKERS)]
        row_id = f"env-extra-{index + 1:05d}"
        prompt = xml_prompt(f"Handle environment blocker: {blocker}.", f"<blocker>{blocker}</blocker><case>{row_id}</case>", "Explain the blocker and the fallback.")
        rows.append(direct_row(prompt, f"Environment blocker: {blocker}. Observation: {observation}. Fallback: {fallback}.", ["fixture", "environment_blocker", "language:en"], meta(row_id, "fixtures", "env-blocker", "training/tests", split=split_for(row_id))))
    return rows
