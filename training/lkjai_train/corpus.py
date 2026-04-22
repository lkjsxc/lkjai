import json
import random
from pathlib import Path

_TOOL_TRAJECTORIES = [
    (
        "List the files in the current directory.",
        "fs.list",
        {"path": "."},
        "README.md\nCargo.toml\nsrc",
        "The current directory contains README.md, Cargo.toml, and a src folder.",
    ),
    (
        "Read the README file.",
        "fs.read",
        {"path": "README.md"},
        "# lkjai\n\nLocal agentic AI system.",
        "The README describes lkjai as a local agentic AI system.",
    ),
    (
        "Show me the contents of src/main.rs.",
        "fs.read",
        {"path": "src/main.rs"},
        'fn main() {\n    println!("hello");\n}',
        "src/main.rs contains a simple Rust main function that prints hello.",
    ),
    (
        "Create a file named notes.txt with 'buy milk'.",
        "fs.write",
        {"path": "notes.txt", "content": "buy milk"},
        "wrote notes.txt",
        "I created notes.txt with the content 'buy milk'.",
    ),
    (
        "Run pwd.",
        "shell.exec",
        {"command": "pwd"},
        "/workspace",
        "The current working directory is /workspace.",
    ),
    (
        "Run ls -la.",
        "shell.exec",
        {"command": "ls -la"},
        "total 12\ndrwxr-xr-x 3 user user 4096 Jan 01 00:00 .",
        "The directory listing shows the expected files and folders.",
    ),
    (
        "Fetch https://example.com.",
        "web.fetch",
        {"url": "https://example.com"},
        "<html><body>Example Domain</body></html>",
        "The page at example.com contains a simple HTML body with 'Example Domain'.",
    ),
    (
        "Remember that I prefer concise answers.",
        "memory.write",
        {"content": "User prefers concise answers."},
        "User prefers concise answers.",
        "Noted. I will keep my answers concise.",
    ),
    (
        "What do you remember about my preferences?",
        "memory.search",
        {"query": "preferences"},
        "User prefers concise answers.",
        "I remember that you prefer concise answers.",
    ),
    (
        "Write a greeting to hello.txt.",
        "fs.write",
        {"path": "hello.txt", "content": "Hello, world!"},
        "wrote hello.txt",
        "I wrote 'Hello, world!' to hello.txt.",
    ),
    (
        "List the src directory.",
        "fs.list",
        {"path": "src"},
        "main.rs\nlib.rs",
        "The src directory contains main.rs and lib.rs.",
    ),
    (
        "Run echo hello.",
        "shell.exec",
        {"command": "echo hello"},
        "hello",
        "The command output is 'hello'.",
    ),
    (
        "Fetch https://github.com.",
        "web.fetch",
        {"url": "https://github.com"},
        "<html>GitHub</html>",
        "GitHub's homepage loaded successfully.",
    ),
    (
        "Remember my name is Alice.",
        "memory.write",
        {"content": "User name is Alice."},
        "User name is Alice.",
        "Got it, Alice.",
    ),
    (
        "Search memory for Alice.",
        "memory.search",
        {"query": "Alice"},
        "User name is Alice.",
        "I found that your name is Alice.",
    ),
]

_DIRECT_ANSWERS = [
    ("What is 2+2?", "2+2 equals 4."),
    ("Say hello.", "Hello!"),
    ("What is Rust?", "Rust is a systems programming language focused on safety and performance."),
    ("Explain Docker.", "Docker is a platform for developing, shipping, and running applications in containers."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Define machine learning.", "Machine learning is a field of AI where systems learn patterns from data."),
]


def generate_corpus(size: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    trajectories = list(_TOOL_TRAJECTORIES)
    direct = list(_DIRECT_ANSWERS)

    while len(rows) < size:
        if rng.random() < 0.3 and direct:
            user, answer = rng.choice(direct)
            rows.append({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps({"kind": "final", "thought": "answer directly", "content": answer})},
                ],
                "tags": ["direct_answer"],
            })
        else:
            user, tool, args, result, final_answer = rng.choice(trajectories)
            rows.append({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps({"kind": "tool_call", "thought": "use tool", "tool": tool, "args": args})},
                    {"role": "tool", "name": tool, "content": result},
                    {"role": "assistant", "content": json.dumps({"kind": "final", "thought": "done", "content": final_answer})},
                ],
                "tags": ["tool_trajectory", tool],
            })
    return rows


def write_corpus(path: Path, size: int = 200, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_corpus(size, seed)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
