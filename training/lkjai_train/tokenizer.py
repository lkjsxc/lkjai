import json
from pathlib import Path


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    pad_id = 258
    vocab_size = 259

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        payload = bytes(i for i in ids if 0 <= i < 256)
        return payload.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "byte", "vocab_size": self.vocab_size}), encoding="utf-8")


def train_tokenizer(paths, vocab_size: int = 32_000) -> Path:
    paths.ensure()
    corpus = paths.raw / "train.txt"
    output = paths.tokenizers / "tokenizer.json"
    try:
        from tokenizers import ByteLevelBPETokenizer
    except Exception:
        ByteTokenizer().save(output)
        return output
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[str(corpus)], vocab_size=vocab_size, special_tokens=["<bos>", "<eos>", "<pad>"])
    tokenizer.save(str(output))
    return output


def load_tokenizer(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "byte":
        return ByteTokenizer()
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def encode(tokenizer, text: str) -> list[int]:
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text).ids
