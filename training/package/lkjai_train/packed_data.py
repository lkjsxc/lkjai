import json
import struct
from array import array
from pathlib import Path

from .formatting import iter_rows
from .objectives import normalize_objective, objective_tokens


class PackedDataset:
    def __init__(self, cache_dir: Path, sequence_len: int, pad_id: int):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner):
                return start_count(cache_dir / "starts.bin")

            def __getitem__(self_inner, index):
                start = read_start(cache_dir / "starts.bin", index)
                ids = read_ids(cache_dir / "tokens.bin", start, sequence_len + 1, pad_id)
                mask = read_mask(cache_dir / "loss_mask.bin", start, sequence_len + 1)
                labels = ids[1:]
                label_mask = mask[1:]
                labels = [label if train else -100 for label, train in zip(labels, label_mask)]
                return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        self._dataset = _Dataset()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


def build_or_load_packed_cache(paths, tokenizer, source: Path, split: str, settings) -> Path:
    objective = normalize_objective(settings.objective)
    eos_id = tokenizer.token_to_id("<eos>") or 0
    cache_dir = paths.datasets / "packed" / cache_name(split, objective, settings.sequence_len)
    meta_path = cache_dir / "metadata.json"
    expected = metadata(source, split, objective, settings, tokenizer.get_vocab_size())
    if cache_is_current(cache_dir, expected):
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_stream_files(cache_dir, tokenizer, source, objective, eos_id)
    ensure_minimum_length(cache_dir / "tokens.bin", cache_dir / "loss_mask.bin", settings.sequence_len + 1, eos_id)
    write_window_starts(cache_dir, settings.sequence_len, objective)
    meta_path.write_text(json.dumps(expected | packed_stats(cache_dir), indent=2), encoding="utf-8")
    return cache_dir


def cache_name(split: str, objective: str, sequence_len: int) -> str:
    return f"{split}-{objective}-seq{sequence_len}"


def metadata(source: Path, split: str, objective: str, settings, vocab_size: int) -> dict:
    if vocab_size > 65535:
        raise ValueError("packed cache v2 requires vocab_size <= 65535")
    return {
        "format": "lkjai-packed-cache-v2",
        "token_dtype": "uint16",
        "source": str(source),
        "source_fingerprint": source_fingerprint(source),
        "split": split,
        "objective": objective,
        "sequence_len": settings.sequence_len,
        "vocab_size": vocab_size,
    }


def source_fingerprint(source: Path) -> dict:
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
    return {
        "files": len([file for file in files if file.exists()]),
        "latest_mtime_ns": max((file.stat().st_mtime_ns for file in files if file.exists()), default=0),
        "total_size": sum(file.stat().st_size for file in files if file.exists()),
    }


def cache_is_current(cache_dir: Path, expected: dict) -> bool:
    meta_path = cache_dir / "metadata.json"
    required = [cache_dir / "tokens.bin", cache_dir / "loss_mask.bin", cache_dir / "starts.bin", meta_path]
    if not all(path.exists() for path in required):
        return False
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(data.get(key) == value for key, value in expected.items())


def write_stream_files(cache_dir: Path, tokenizer, source: Path, objective: str, eos_id: int) -> None:
    token_buffer = array("H")
    mask_buffer = array("B")
    with (cache_dir / "tokens.bin").open("wb") as token_file, (cache_dir / "loss_mask.bin").open("wb") as mask_file:
        for row in iter_rows(source):
            item = objective_tokens(tokenizer, row, objective)
            token_buffer.extend(item.ids)
            token_buffer.append(eos_id)
            mask_buffer.extend(item.loss_mask)
            mask_buffer.append(0)
            if len(token_buffer) >= 1_000_000:
                token_buffer.tofile(token_file)
                mask_buffer.tofile(mask_file)
                token_buffer = array("H")
                mask_buffer = array("B")
        if token_buffer:
            token_buffer.tofile(token_file)
            mask_buffer.tofile(mask_file)


def ensure_minimum_length(tokens_path: Path, masks_path: Path, minimum_tokens: int, eos_id: int) -> None:
    current = token_count(tokens_path)
    if current >= minimum_tokens:
        return
    missing = minimum_tokens - current
    with tokens_path.open("ab") as token_file, masks_path.open("ab") as mask_file:
        array("H", [eos_id] * missing).tofile(token_file)
        array("B", [0] * missing).tofile(mask_file)


def write_window_starts(cache_dir: Path, sequence_len: int, objective: str) -> None:
    tokens = token_count(cache_dir / "tokens.bin")
    starts = array("Q")
    with (cache_dir / "starts.bin").open("wb") as file, (cache_dir / "loss_mask.bin").open("rb") as mask_file:
        for start in range(0, max(1, tokens - 1), sequence_len):
            if objective == "assistant_masked_sft":
                mask_file.seek(start + 1)
                if not any(mask_file.read(sequence_len)):
                    continue
            starts.append(start)
            if len(starts) >= 1_000_000:
                starts.tofile(file)
                starts = array("Q")
        if starts:
            starts.tofile(file)
    if start_count(cache_dir / "starts.bin") == 0:
        with (cache_dir / "starts.bin").open("wb") as file:
            array("Q", [0]).tofile(file)


def packed_stats(cache_dir: Path) -> dict:
    tokens = token_count(cache_dir / "tokens.bin")
    return {
        "tokens": tokens,
        "loss_tokens": count_loss_tokens(cache_dir / "loss_mask.bin"),
        "windows": start_count(cache_dir / "starts.bin"),
    }


def count_loss_tokens(path: Path) -> int:
    total = 0
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            total += sum(chunk)
    return total


def token_count(path: Path) -> int:
    return path.stat().st_size // 2 if path.exists() else 0


def start_count(path: Path) -> int:
    return path.stat().st_size // 8 if path.exists() else 0


def read_start(path: Path, index: int) -> int:
    with path.open("rb") as file:
        file.seek(index * 8)
        return struct.unpack("<Q", file.read(8))[0]


def read_ids(path: Path, start: int, count: int, pad_id: int) -> list[int]:
    with path.open("rb") as file:
        file.seek(start * 2)
        data = file.read(count * 2)
    items = array("H")
    items.frombytes(data[: len(data) - (len(data) % 2)])
    values = list(items)
    if len(values) < count:
        values.extend([pad_id] * (count - len(values)))
    return values


def read_mask(path: Path, start: int, count: int) -> list[int]:
    with path.open("rb") as file:
        file.seek(start)
        data = file.read(count)
    items = array("B")
    items.frombytes(data)
    values = list(items)
    if len(values) < count:
        values.extend([0] * (count - len(values)))
    return values
