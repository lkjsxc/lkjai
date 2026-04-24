from pathlib import Path

from .corpus_shared import split_for, xml_prompt
from .rows import action_message, kimi_meta, multi_turn_row


SOURCE_FILES = [
    "src/config.rs", "src/lib.rs", "src/main.rs", "src/model_client.rs",
    "src/agent/schema.rs", "src/agent/action.rs", "src/agent/tools.rs",
    "Cargo.toml", "docker-compose.yml", "verify.sh",
]

FAILURE_OBSERVATIONS = [
    "error: test failed",
    "error: could not compile",
    "thread panicked",
    "assertion failed",
]


def kimi_impl_rows(limit: int) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    rows = []
    for index in range(limit):
        rel = SOURCE_FILES[index % len(SOURCE_FILES)]
        path = root / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("//")][:20]
        snippet = " ".join(lines[:8])[:360]
        row_id = f"kimi-impl-{index + 1:06d}"
        task = f"Debug and fix an issue related to {rel}."
        prompt = xml_prompt(task, f"<path>{rel}</path><snippet>{snippet[:180]}</snippet>", "Use tools before answering.")
        messages = [
            {"role": "user", "content": prompt},
            action_message({"kind": "plan", "content": f"List files, read {rel}, check tests, identify failure, plan fix."}),
            action_message({"kind": "tool_call", "thought": "list source dir", "tool": "fs.list", "args": {"path": str(Path(rel).parent) or "."}}),
            {"role": "tool", "name": "fs.list", "content": "\n".join(SOURCE_FILES[:6])},
            action_message({"kind": "tool_call", "thought": f"read {rel}", "tool": "fs.read", "args": {"path": rel}}),
            {"role": "tool", "name": "fs.read", "content": snippet[:320]},
            action_message({"kind": "tool_call", "thought": "run tests", "tool": "shell.exec", "args": {"command": f"cargo test --lib {Path(rel).stem}", "timeout": 60}}),
            {"role": "tool", "name": "shell.exec", "content": FAILURE_OBSERVATIONS[index % len(FAILURE_OBSERVATIONS)]},
            action_message({"kind": "tool_call", "thought": "read related file", "tool": "fs.read", "args": {"path": "src/lib.rs"}}),
            {"role": "tool", "name": "fs.read", "content": "pub mod config; pub mod model_client;"},
            action_message({"kind": "final", "content": f"Identified issue in {rel}. The test failed with {FAILURE_OBSERVATIONS[index % len(FAILURE_OBSERVATIONS)]}. Review the source and rerun tests after fix."}),
        ]
        rows.append(multi_turn_row(messages, ["kimi_generated", "implementation", "debugging", "multi_turn", "language:en"], kimi_meta(row_id, "source-code", "debugging", rel, split=split_for(row_id))))
    return rows
