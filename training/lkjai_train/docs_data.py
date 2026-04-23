from pathlib import Path

from .rows import final_row


def docs_rows(root: Path, limit: int = 60) -> list[dict]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*.md")):
        if len(rows) >= limit:
            break
        text = path.read_text(encoding="utf-8")
        title = first_heading(text) or path.stem.replace("-", " ")
        snippet = compact(text)
        if not snippet:
            continue
        prompt = f"What does {path.as_posix()} define?"
        rows.append(final_row(prompt, f"{title}: {snippet}", ["docs_grounding", path.as_posix()]))
    return rows


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def compact(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped.lstrip("- "))
        if len(" ".join(lines)) > 240:
            break
    return " ".join(lines)[:300]
