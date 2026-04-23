from pathlib import Path


class Paths:
    def __init__(self, data_dir: str = "/app/data") -> None:
        self.root = Path(data_dir)
        self.datasets = self.root / "datasets"
        self.tokenizer = self.root / "tokenizer"
        self.checkpoints = self.root / "checkpoints"
        self.exports = self.root / "exports"
        self.runs = self.root / "runs"
        self.preferences = self.root / "preferences"
        self.fixtures = self.datasets / "fixtures.jsonl"
        self.corpus = self.datasets / "corpus.jsonl"
        self.train_dataset = self.datasets / "train.jsonl"
        self.val_dataset = self.datasets / "val.jsonl"
        self.holdout_dataset = self.datasets / "holdout.jsonl"
        self.dataset_metadata = self.datasets / "metadata.json"
        self.tokenizer_json = self.tokenizer / "tokenizer.json"
        self.tokenizer_manifest = self.tokenizer / "manifest.json"
        self.training_summary = self.checkpoints / "training-summary.json"
        self.checkpoint_final = self.checkpoints / "final"
        self.checkpoint_best = self.checkpoints / "best"
        self.checkpoint_dpo = self.checkpoints / "dpo"
        self.dpo_summary = self.checkpoints / "dpo-summary.json"
        self.checkpoint_manifest = self.checkpoints / "manifest.json"
        self.export_manifest = self.exports / "manifest.json"
        self.preference_pairs = self.preferences / "pairs.jsonl"

    def ensure(self) -> None:
        for path in [self.datasets, self.tokenizer, self.checkpoints, self.exports, self.runs, self.preferences]:
            path.mkdir(parents=True, exist_ok=True)
