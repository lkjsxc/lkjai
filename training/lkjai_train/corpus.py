import json
from itertools import islice
from pathlib import Path

FIXTURE = [
    "Small language models need clean text, stable tokenization, and careful tests.",
    "An agent can run commands, fetch websites, and edit files when tools are logged.",
    "CUDA training benefits from packed token arrays and resumable checkpoints.",
    "The final serving artifact must stay below five hundred twelve mebibytes.",
]


def prepare_corpus(paths, token_budget: int, dataset: str, tiny: bool = False) -> Path:
    paths.ensure()
    out = paths.raw / "train.txt"
    rows = FIXTURE if tiny else stream_dataset(dataset, token_budget)
    tokens = 0
    count = 0
    with out.open("w", encoding="utf-8") as file:
        for text in rows:
            if not text:
                continue
            file.write(text.replace("\n", " ").strip() + "\n")
            tokens += max(1, len(text.split()))
            count += 1
            if tokens >= token_budget:
                break
    metadata = {
        "dataset": dataset,
        "token_budget": token_budget,
        "estimated_tokens": tokens,
        "rows": count,
        "license_note": "Preserve upstream FineWeb-Edu metadata for full runs.",
    }
    (paths.raw / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def stream_dataset(dataset: str, token_budget: int):
    try:
        from datasets import load_dataset
    except Exception:
        repeats = max(1, token_budget // 40)
        return islice((line for _ in range(repeats) for line in FIXTURE), repeats * len(FIXTURE))
    rows = load_dataset(dataset, split="train", streaming=True)
    return (row.get("text", "") for row in rows)
