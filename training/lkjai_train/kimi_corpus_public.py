from .corpus_shared import split_for, xml_prompt
from .public_import import ALLOWED_LICENSES, public_sources
from .rows import direct_row, kimi_meta


def kimi_public_rows(limit: int) -> list[dict]:
    rows = []
    try:
        entries = public_sources()
    except Exception:
        entries = []
    if not entries:
        return rows
    for index, entry in enumerate(entries):
        if len(rows) >= limit:
            break
        text = entry.get("text", "")
        license_name = entry.get("license", "")
        if license_name not in ALLOWED_LICENSES:
            continue
        row_id = f"kimi-public-{index + 1:06d}"
        prompt = xml_prompt("Summarize the imported text.", "<source>public-import</source>", "Keep it concise.")
        rows.append(direct_row(prompt, text[:400], ["kimi_generated", "public_import", "language:en"], kimi_meta(row_id, "public-import", "summarize", entry.get("source_ref", "public-import"), split=split_for(row_id), license_name=license_name)))
    return rows
