import json
from pathlib import Path


def train_policy(paths) -> Path:
    paths.ensure()
    rows = [json.loads(line) for line in paths.fixtures.read_text(encoding="utf-8").splitlines()]
    rules = base_rules()
    tags = sorted({tag for row in rows for tag in row.get("tags", [])})
    policy = {
        "format": "lkjai-trained-policy-v1",
        "trained_on": str(paths.fixtures),
        "rows": len(rows),
        "tags": tags,
        "rules": rules,
        "fallback": {
            "kind": "final",
            "thought": "answer directly",
            "content": "I am running from the trained local lkjai policy.",
        },
    }
    paths.policy_model.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return paths.policy_model


def base_rules() -> list[dict]:
    return [
        {
            "any": ["remember", "prefer"],
            "action": {
                "kind": "tool_call",
                "thought": "store durable user memory",
                "tool": "memory.write",
                "args": {"content": "{{memory}}"},
            },
        },
        {
            "any": ["list", "directory", "files"],
            "action": {
                "kind": "tool_call",
                "thought": "inspect directory",
                "tool": "fs.list",
                "args": {"path": "{{path}}"},
            },
        },
        {
            "any": ["read"],
            "action": {
                "kind": "tool_call",
                "thought": "read requested file",
                "tool": "fs.read",
                "args": {"path": "{{path}}"},
            },
        },
        {
            "any": ["fetch", "http://", "https://"],
            "action": {
                "kind": "tool_call",
                "thought": "fetch requested web page",
                "tool": "web.fetch",
                "args": {"url": "{{url}}"},
            },
        },
    ]
