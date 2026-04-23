import random
from collections import Counter

from .corpus_docs import doc_rows
from .corpus_kjxlkj import kjxlkj_rows
from .corpus_synthetic import general_rows, local_tool_rows, safety_rows
from .kjxlkj_data import VISIBILITY_RULES
from .public_data import PUBLIC_SOURCE_METADATA
from .rows import signature


def generate_corpus(size: int = 12000, seed: int = 42) -> list[dict]:
    rows = doc_rows(1800) + general_rows(8000) + safety_rows(2000, VISIBILITY_RULES) + kjxlkj_rows(1700) + local_tool_rows(1500)
    rows = dedupe_rows(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) < size:
        raise RuntimeError(f"corpus generator produced only {len(rows)} rows")
    return rows[:size]


def split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    split = {"train": [], "val": [], "holdout": []}
    for row in rows:
        split[row["meta"]["split"]].append(row)
    return split


def source_metadata(rows: list[dict]) -> list[dict]:
    counts = Counter((row["meta"]["domain"], row["meta"]["license"]) for row in rows)
    metadata = [{"name": name, "license": license_name, "rows": count} for (name, license_name), count in sorted(counts.items())]
    return metadata + PUBLIC_SOURCE_METADATA


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = signature(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique
