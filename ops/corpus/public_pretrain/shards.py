import json
from pathlib import Path


class ShardWriter:
    def __init__(self, root, split, rows_per_shard=100000):
        self.root = Path(root) / split
        self.split = split
        self.rows_per_shard = rows_per_shard
        self.index = 0
        self.rows = 0
        self.handle = None

    def write(self, row):
        if self.handle is None or self.rows >= self.rows_per_shard:
            self.rotate()
        self.handle.write(json.dumps(row, ensure_ascii=False,
                                     separators=(",", ":")))
        self.handle.write("\n")
        self.rows += 1

    def rotate(self):
        if self.handle:
            self.handle.close()
        self.index += 1
        self.rows = 0
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"{self.split}-{self.index:06d}.jsonl"
        self.handle = (self.root / name).open("w", encoding="utf-8")

    def close(self):
        if self.handle:
            self.handle.close()


def clean_output(out_dir):
    for split in ("train", "val", "holdout"):
        split_dir = Path(out_dir) / split
        if split_dir.exists():
            for path in split_dir.glob("*.jsonl"):
                path.unlink()
