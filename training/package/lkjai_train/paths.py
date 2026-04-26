import os
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
        self.public_corpus = self.datasets / "public.jsonl"
        self.public_manifest = self.datasets / "public-sources.json"
        self.train_dataset = self.datasets / "train.jsonl"
        self.val_dataset = self.datasets / "val.jsonl"
        self.holdout_dataset = self.datasets / "holdout.jsonl"
        self.dataset_metadata = self.datasets / "metadata.json"
        self.tokenizer_json = self.tokenizer / "tokenizer.json"
        self.tokenizer_manifest = self.tokenizer / "manifest.json"
        self.training_summary = self.checkpoints / "training-summary.json"
        self.checkpoint_final = self.checkpoints / "final"
        self.checkpoint_best = self.checkpoints / "best"
        self.checkpoint_latest = self.checkpoints / "latest"
        self.checkpoint_steps = self.checkpoints / "steps"
        self.checkpoint_dpo = self.checkpoints / "dpo"
        self.dpo_summary = self.checkpoints / "dpo-summary.json"
        self.checkpoint_simpo = self.checkpoints / "simpo"
        self.simpo_summary = self.checkpoints / "simpo-summary.json"
        self.checkpoint_manifest = self.checkpoints / "manifest.json"
        self.export_manifest = self.exports / "manifest.json"
        self.preference_pairs = self.preferences / "pairs.jsonl"
        self.kimi_corpus = self.root / "kimi-corpus"
        self.kimi_train = self.kimi_corpus / "train" / "train-0001.jsonl"
        self.kimi_val = self.kimi_corpus / "val" / "val-0001.jsonl"
        self.kimi_holdout = self.kimi_corpus / "holdout" / "holdout-0001.jsonl"
        self.kimi_manifest = self.kimi_corpus / "manifest.json"
        self.kimi_validation_report = self.kimi_corpus / "validation-report.json"
        default_committed = "/workspace/corpus/generated/kimi-full-v1"
        self.committed_kimi_corpus = Path(os.environ.get("TRAIN_COMMITTED_CORPUS_DIR", default_committed))
        self.committed_train = self.committed_kimi_corpus / "train"
        self.committed_val = self.committed_kimi_corpus / "val"
        self.committed_holdout = self.committed_kimi_corpus / "holdout"

    def ensure(self) -> None:
        for path in [self.datasets, self.tokenizer, self.checkpoints, self.exports, self.runs, self.preferences]:
            path.mkdir(parents=True, exist_ok=True)
