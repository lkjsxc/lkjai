import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .formatting import SPECIAL_TOKENS, load_rows, row_text


def train_tokenizer(paths, settings) -> Path:
    paths.ensure()
    dataset = paths.corpus if paths.corpus.exists() else paths.fixtures
    if not dataset.exists():
        raise RuntimeError(f"tokenizer dataset not found: {dataset}")
    texts = [row_text(row) for row in load_rows(dataset)]
    if not texts:
        raise RuntimeError("tokenizer training requires at least one row")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=settings.vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(paths.tokenizer_json))
    manifest = {
        "format": "lkjai-tokenizer-manifest-v1",
        "kind": "byte-bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": str(dataset),
        "special_tokens": SPECIAL_TOKENS,
    }
    paths.tokenizer_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.tokenizer_json


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise RuntimeError(f"tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))
