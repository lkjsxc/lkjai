import json
from pathlib import Path


def source_paths(root: Path | None, source: dict) -> list[Path]:
    if root is None:
        return []
    if source.get("local_file"):
        path = root / source["local_file"]
        return [path] if path.is_file() else []
    pattern = source.get("local_glob", "")
    if pattern:
        return [path for path in sorted(root.glob(pattern)) if path.is_file()]
    return []


def iter_source_text(path: Path, text_field: str):
    if path.suffix == ".jsonl":
        yield from iter_jsonl_text(path, text_field)
        return
    if path.suffix == ".parquet":
        yield from iter_parquet_text(path, text_field)
        return
    raise ValueError(f"unsupported public pretrain file format: {path}")


def iter_jsonl_text(path: Path, text_field: str):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get(text_field) if isinstance(item, dict) else None
            if isinstance(text, str) and usable_english(text):
                yield text.strip()


def iter_parquet_text(path: Path, text_field: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow or use the train Docker image to read Cosmopedia Parquet files") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=[text_field]):
        column = batch.column(text_field)
        for value in column.to_pylist():
            if isinstance(value, str) and usable_english(value):
                yield value.strip()


def usable_english(text: str) -> bool:
    if len(text.strip()) < 200:
        return False
    non_ascii_letters = sum(1 for char in text if ord(char) > 127 and char.isalpha())
    return non_ascii_letters / max(1, len(text)) < 0.03
