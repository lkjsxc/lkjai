import random

from .corpus import dedupe_rows, source_metadata
from .kimi_corpus_confirm import kimi_confirm_rows
from .kimi_corpus_docs import kimi_doc_rows
from .kimi_corpus_everyday import kimi_everyday_rows
from .kimi_corpus_impl import kimi_impl_rows
from .kimi_corpus_recovery import kimi_recovery_rows
from .kimi_corpus_schema import kimi_schema_rows
from .rows import signature


MIX = {
    "everyday": 0.58,
    "schema": 0.12,
    "recovery": 0.10,
    "docs": 0.08,
    "impl": 0.07,
    "confirm": 0.04,
    "extra_docs": 0.01,
}


def generate_kimi_corpus(row_target: int = 5000, seed: int = 42) -> list[dict]:
    rows = []
    rows += kimi_everyday_rows(int(row_target * MIX["everyday"]) + 1)
    rows += kimi_doc_rows(int(row_target * MIX["docs"]) + 1)
    rows += kimi_impl_rows(int(row_target * MIX["impl"]) + 1)
    rows += kimi_schema_rows(int(row_target * MIX["schema"]) + 1)
    rows += kimi_recovery_rows(int(row_target * MIX["recovery"]) + 1)
    rows += kimi_confirm_rows(int(row_target * MIX["confirm"]) + 1)
    rows += kimi_doc_rows(int(row_target * MIX["extra_docs"]) + 1)
    rows = dedupe_rows(rows)
    if len(rows) < row_target:
        rows = dedupe_rows(rows + kimi_everyday_rows(row_target * 2))
    random.Random(seed).shuffle(rows)
    if len(rows) > row_target:
        rows = rows[:row_target]
    return rows


def kimi_source_metadata(rows: list[dict]) -> list[dict]:
    return source_metadata(rows)
