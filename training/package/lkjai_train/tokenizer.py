import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from .formatting import SPECIAL_TOKENS, iter_rows, row_text
from .tokenizer_tokens import BASE_SPECIAL_TOKENS, XML_TAG_TOKENS, bpe_vocab_size


def train_tokenizer(paths, settings) -> Path:
    paths.ensure()
    dataset = train_source(paths, settings.objective)
    if not any(iter_rows(dataset)):
        raise RuntimeError("tokenizer training requires at least one row")
    tokenizer = train_text_tokenizer(
        (row_text(row) for row in iter_rows(dataset)),
        settings.vocab_size,
    )
    tokenizer.save(str(paths.tokenizer_json))
    train_rows, train_tokens = 0, 0
    for row in iter_rows(dataset):
        train_rows += 1
        train_tokens += len(tokenizer.encode(row_text(row)).ids)
    manifest = {
        "format": "lkjai-tokenizer-manifest-v2",
        "kind": "byte-bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "configured_vocab_size": settings.vocab_size,
        "dataset": str(dataset),
        "special_tokens": SPECIAL_TOKENS,
        "base_special_tokens": BASE_SPECIAL_TOKENS,
        "xml_tag_tokens": XML_TAG_TOKENS,
        "train_rows": train_rows,
        "train_tokens": train_tokens,
    }
    paths.tokenizer_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths.tokenizer_json


def train_source(paths, objective: str = "causal_lm_full") -> Path:
    if objective in {"assistant_masked_sft", "sft", "assistant", "supervised"}:
        if paths.committed_train.exists() and any(paths.committed_train.rglob("*.jsonl")):
            return paths.committed_train
        if paths.train_dataset.exists():
            return paths.train_dataset
    if paths.public_pretrain_train.exists() and any(paths.public_pretrain_train.rglob("*.jsonl")):
        return paths.public_pretrain_train
    if paths.train_dataset.exists():
        return paths.train_dataset
    if paths.committed_train.exists() and any(paths.committed_train.rglob("*.jsonl")):
        return paths.committed_train
    return paths.corpus if paths.corpus.exists() else paths.fixtures


def train_text_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tokenizer = base_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=bpe_vocab_size(vocab_size),
        min_frequency=1,
        special_tokens=BASE_SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    add_xml_tag_tokens(tokenizer)
    return tokenizer


def base_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def add_xml_tag_tokens(tokenizer: Tokenizer) -> None:
    tokens = [
        AddedToken(token, special=False, normalized=False)
        for token in XML_TAG_TOKENS
    ]
    tokenizer.add_tokens(tokens)


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise RuntimeError(f"tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))
