from pathlib import Path


class Paths:
    def __init__(self, data_dir: str = "/app/data") -> None:
        self.root = Path(data_dir)
        self.datasets = self.root / "datasets"
        self.adapters = self.root / "adapters"
        self.exports = self.root / "exports"
        self.policy = self.root / "policy"
        self.runs = self.root / "runs"
        self.fixtures = self.datasets / "fixtures.jsonl"
        self.dataset_metadata = self.datasets / "metadata.json"
        self.adapter_manifest = self.adapters / "manifest.json"
        self.export_manifest = self.exports / "manifest.json"
        self.policy_model = self.policy / "model.json"

    def ensure(self) -> None:
        for path in [self.datasets, self.adapters, self.exports, self.policy, self.runs]:
            path.mkdir(parents=True, exist_ok=True)
