import json
from pathlib import Path

import numpy as np

from .tokenizer import encode, load_tokenizer


def corpus_lines(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n")


def pack_tokens(paths) -> Path:
    paths.ensure()
    tokenizer = load_tokenizer(paths.tokenizers / "tokenizer.json")
    out = paths.tokenized / "tokens.u16"
    tokens = 0
    max_token_id = 0
    uint16_max = int(np.iinfo(np.uint16).max)
    with out.open("wb") as file:
        for line in corpus_lines(paths.raw / "train.txt"):
            encoded = encode(tokenizer, line)
            if not encoded:
                continue
            local_max = max(encoded)
            if local_max > uint16_max:
                raise ValueError(f"token id {local_max} exceeds uint16 capacity")
            np.asarray(encoded, dtype=np.uint16).tofile(file)
            tokens += len(encoded)
            max_token_id = max(max_token_id, local_max)
    metadata = {
        "tokens": tokens,
        "dtype": "uint16",
        "token_file": out.name,
        "format": "flat-binary",
        "max_token_id": max_token_id,
    }
    (paths.tokenized / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out
