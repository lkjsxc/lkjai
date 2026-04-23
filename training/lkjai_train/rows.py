import json


def direct_row(user: str, answer: str, tags: list[str] | None = None) -> dict:
    action = {"kind": "final", "thought": "answer directly", "content": answer}
    return row([{"role": "user", "content": user}, assistant(action)], tags or ["direct_answer"])


def final_row(user: str, answer: str, tags: list[str]) -> dict:
    action = {"kind": "final", "thought": "answer from project docs", "content": answer}
    return row([{"role": "user", "content": user}, assistant(action)], tags)


def tool_row(user: str, tool: str, args: dict, result: str, final_answer: str) -> dict:
    action = {"kind": "tool_call", "thought": "use tool", "tool": tool, "args": args}
    final = {"kind": "final", "thought": "done", "content": final_answer}
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
    return {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}


def row(messages: list[dict], tags: list[str]) -> dict:
    return {"messages": messages, "tags": tags}
