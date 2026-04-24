import random
from collections import Counter

from .corpus_docs import doc_rows
from .corpus_synthetic import fixture_rows, repo_schema_rows
from .rows import signature


def generate_corpus(size: int = 6000, seed: int = 42) -> list[dict]:
    rows = doc_rows(4200) + repo_schema_rows(1200) + fixture_rows(900)
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
    return [{"name": name, "license": license_name, "rows": count} for (name, license_name), count in sorted(counts.items())]


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = signature(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique
