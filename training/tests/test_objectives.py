import json
from types import SimpleNamespace

from lkjai_train.objectives import ASSISTANT_MASKED_SFT, CAUSAL_LM_FULL, objective_tokens
from lkjai_train.packed_data import build_or_load_packed_cache, read_ids, read_mask, start_count
from lkjai_train.paths import Paths


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
    assert read_ids(cache / "tokens.bin", 0, 5, 0) == [97, 98, 99, 100, 101]
    assert read_mask(cache / "loss_mask.bin", 1, 4) == [1, 1, 1, 1]


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
