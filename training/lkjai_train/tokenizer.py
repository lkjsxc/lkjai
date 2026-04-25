import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .formatting import SPECIAL_TOKENS, iter_rows, row_text


def train_tokenizer(paths, settings) -> Path:
    paths.ensure()
    dataset = train_source(paths)
    if not any(iter_rows(dataset)):
        raise RuntimeError("tokenizer training requires at least one row")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=settings.vocab_size, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator((row_text(row) for row in iter_rows(dataset)), trainer=trainer)
    tokenizer.save(str(paths.tokenizer_json))
    train_rows, train_tokens = 0, 0
    for row in iter_rows(dataset):
        train_rows += 1
        train_tokens += len(tokenizer.encode(row_text(row)).ids)
    manifest = {
        "format": "lkjai-tokenizer-manifest-v2",
        "kind": "byte-bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": str(dataset),
        "special_tokens": SPECIAL_TOKENS,
        "train_rows": train_rows,
        "train_tokens": train_tokens,
    }
    paths.tokenizer_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.tokenizer_json


def train_source(paths) -> Path:
    if paths.committed_train.exists() and any(paths.committed_train.rglob("*.jsonl")):
        return paths.committed_train
    if paths.train_dataset.exists():
        return paths.train_dataset
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise RuntimeError(f"tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))
