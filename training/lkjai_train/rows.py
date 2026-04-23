import json


def direct_row(user: str, answer: str, tags: list[str] | None = None) -> dict:
    action = {"kind": "final", "content": answer}
    return row([{"role": "user", "content": user}, assistant(action)], tags or ["direct_answer"])


def final_row(user: str, answer: str, tags: list[str]) -> dict:
    action = {"kind": "final", "content": answer}
    return row([{"role": "user", "content": user}, assistant(action)], tags)


def tool_row(user: str, tool: str, args: dict, result: str, final_answer: str) -> dict:
    action = {"kind": "tool_call", "tool": tool, "args": args}
    final = {"kind": "final", "content": final_answer}
    return row(
        [
            {"role": "user", "content": user},
            assistant(action),
            {"role": "tool", "name": tool, "content": result},
            assistant(final),
        ],
        ["tool_trajectory", tool],
    )


def assistant(action: dict) -> dict:
    text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    return {"role": "assistant", "content": text}


def row(messages: list[dict], tags: list[str]) -> dict:
    return {"messages": messages, "tags": tags}
