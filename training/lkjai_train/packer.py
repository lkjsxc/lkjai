import json
from pathlib import Path

import numpy as np

from .tokenizer import encode, load_tokenizer


def pack_tokens(paths) -> Path:
    paths.ensure()
    tokenizer = load_tokenizer(paths.tokenizers / "tokenizer.json")
    ids: list[int] = []
    for line in (paths.raw / "train.txt").read_text(encoding="utf-8").splitlines():
        ids.extend(encode(tokenizer, line))
    array = np.asarray(ids, dtype=np.uint16)
    out = paths.tokenized / "tokens.npy"
    np.save(out, array)
    metadata = {"tokens": int(array.size), "dtype": "uint16"}
    (paths.tokenized / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out
