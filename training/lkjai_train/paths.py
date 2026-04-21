from pathlib import Path


class Paths:
    def __init__(self, data_dir: str = "/app/data") -> None:
        self.root = Path(data_dir)
        self.raw = self.root / "corpus" / "raw"
        self.tokenized = self.root / "corpus" / "tokenized"
        self.tokenizers = self.root / "tokenizers"
        self.checkpoints = self.root / "checkpoints"
        self.models = self.root / "models"
        self.runs = self.root / "runs"

    def ensure(self) -> None:
        for path in [
            self.raw,
            self.tokenized,
            self.tokenizers,
            self.checkpoints,
            self.models,
            self.runs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
