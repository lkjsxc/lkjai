from pathlib import Path


def doc_records(root: Path) -> list[dict]:
    if not root.exists():
        return []
    records = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        display_path = path.relative_to(root.parent).as_posix()
        title = first_heading(text) or path.stem.replace("-", " ")
        snippet = compact(text)
        if not snippet:
            continue
        records.append(
            {
                "path": display_path,
                "title": title,
                "snippet": snippet,
                "defaults": defaults_line(text),
                "verification": verification_line(text),
            }
        )
    return records


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
        if len(" ".join(lines)) > 360:
            break
    return " ".join(lines)[:420]


def defaults_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("-") and "default" in line.lower():
            return line.strip().lstrip("- ").strip()
    return ""


def verification_line(text: str) -> str:
    capture = False
    for line in text.splitlines():
        if line.startswith("## Verification"):
            capture = True
            continue
        if capture and line.strip():
            return line.strip().lstrip("- ").strip()
    return ""
