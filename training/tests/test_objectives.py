import json
from types import SimpleNamespace

from lkjai_train.objectives import ASSISTANT_MASKED_SFT, CAUSAL_LM_FULL, objective_tokens
from lkjai_train.packed_data import build_or_load_packed_cache, read_ids, read_mask, start_count
from lkjai_train.paths import Paths
from lkjai_train.scratch_loaders import train_source as loader_train_source


def test_full_lm_objective_marks_all_tokens_for_loss():
    item = objective_tokens(DummyTokenizer(), {"text": "abc"}, CAUSAL_LM_FULL)
    assert item.ids == [97, 98, 99]
    assert item.loss_mask == [1, 1, 1]


def test_assistant_masked_objective_preserves_non_assistant_masking():
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>hi</content></action>"},
        ]
    }
    item = objective_tokens(DummyTokenizer(), row, ASSISTANT_MASKED_SFT)
    assert len(item.ids) == len(item.loss_mask)
    assert 0 in item.loss_mask
    assert 1 in item.loss_mask


def test_packed_cache_writes_lazy_window_inputs_and_masks(tmp_path):
    paths = Paths(str(tmp_path))
    paths.ensure()
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps({"text": "abcdef"}) + "\n", encoding="utf-8")
    settings = SimpleNamespace(objective=CAUSAL_LM_FULL, sequence_len=4)
    cache = build_or_load_packed_cache(paths, DummyTokenizer(), source, "train", settings)
    assert start_count(cache / "starts.bin") == 2
    meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    assert meta["format"] == "lkjai-packed-cache-v2"
    assert meta["token_dtype"] == "uint16"
    assert (cache / "tokens.bin").stat().st_size == 14
    assert read_ids(cache / "tokens.bin", 0, 5, 0) == [97, 98, 99, 100, 101]
    assert read_mask(cache / "loss_mask.bin", 1, 4) == [1, 1, 1, 1]


def test_sft_training_source_prefers_xml_rows_over_public_pretrain(tmp_path, monkeypatch):
    committed = tmp_path / "committed"
    monkeypatch.setenv("TRAIN_COMMITTED_CORPUS_DIR", str(committed))
    paths = Paths(str(tmp_path / "data"))
    public_train = paths.public_pretrain_train / "train-000001.jsonl"
    public_train.parent.mkdir(parents=True)
    public_train.write_text(json.dumps({"mode": "pretrain", "text": "public text"}) + "\n", encoding="utf-8")
    committed_train = paths.committed_train / "train-000001.jsonl"
    committed_train.parent.mkdir(parents=True)
    committed_train.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    assert loader_train_source(paths, CAUSAL_LM_FULL) == paths.public_pretrain_train
    assert loader_train_source(paths, ASSISTANT_MASKED_SFT) == paths.committed_train


class DummyTokenizer:
    def encode(self, text):
        return Encoded([ord(char) for char in text])

    def token_to_id(self, token):
        return 0 if token in {"<eos>", "<pad>"} else None

    def get_vocab_size(self):
        return 256


class Encoded:
    def __init__(self, ids):
        self.ids = ids
