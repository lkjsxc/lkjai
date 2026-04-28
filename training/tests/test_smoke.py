import json

import pytest

from lkjai_train.cli import dispatch, train_settings
from lkjai_train.corpus import generate_corpus
from lkjai_train.corpus_source import load_entries, tagged_contents, validate_sources
from lkjai_train.dataset import parse_assistant_xml, prepare_fixtures, validate_dataset
from lkjai_train.behavioral import action_schema_error, bucket, bucket_rates
from lkjai_train.formatting import message_text, prompt_text, supervised_token_ids
from lkjai_train.generation import agent_context_messages, first_xml_action, latest_user_event, normalize_action, normalize_messages
from lkjai_train.paths import Paths
from lkjai_train.preference import prepare_preferences
from lkjai_train.public_import import ALLOWED_LICENSES, prepare_public_corpus, validate_public_sources


torch = None
tokenizers = None
try:
    import torch
    import tokenizers
except ImportError:
    pass


class Args:
    command = "smoke"


@pytest.mark.slow
@pytest.mark.skipif(torch is None or tokenizers is None, reason="training deps not installed")
def test_smoke_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_MAX_STEPS", "1")
    result = dispatch(Args(), Paths(str(tmp_path)))
    assert result.name == "fixed-eval.json"
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["pass_rate"] == 1.0
    assert (tmp_path / "checkpoints" / "manifest.json").exists()
    assert (tmp_path / "checkpoints" / "final" / "model.pt").exists()
    assert (tmp_path / "checkpoints" / "latest" / "model.pt").exists()
    manifest = json.loads((tmp_path / "checkpoints" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["latest_checkpoint_dir"]
    assert "retained_intermediate_checkpoints" in manifest


def test_agent_settings_defaults(monkeypatch):
    monkeypatch.delenv("TRAIN_CONFIG", raising=False)
    monkeypatch.delenv("TRAIN_MODEL_PRESET", raising=False)
    monkeypatch.delenv("TRAIN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    settings = train_settings("agent")
    assert settings.model_preset == "scratch-40m"
    assert settings.model_name == "lkjai-scratch-40m"
    assert settings.objective == "causal_lm_full"
    assert settings.sequence_len == 1024
    assert settings.hidden_size == 576
    assert settings.layers == 10
    assert settings.kv_heads == 2
    assert settings.batch_size == 2
    assert settings.gradient_accumulation == 4
    assert settings.dataloader_impl == "mapped"
    assert settings.batch_policy == "oom_fallback"
    assert settings.activation_checkpoint == "every_n"
    assert settings.compile == "auto"
    assert settings.auto_batch is True
    assert settings.max_optimizer_steps == 400000
    assert settings.save_latest_every_optimizer_steps == 3000
    assert settings.intermediate_save_every_optimizer_steps == 120000
    assert settings.keep_last_checkpoints == 8
    assert settings.corpus_size == 120000
    assert settings.behavioral_threshold == 0.35


def test_quick_settings_are_tiny():
    settings = train_settings("quick")
    assert settings.sequence_len == 64
    assert settings.hidden_size == 64
    assert settings.max_steps == 5
    assert settings.max_optimizer_steps == 5
    assert settings.batch_policy == "fixed"
    assert settings.save_latest_every_optimizer_steps == 1


def test_source_corpus_files_are_tagged_json_arrays():
    validate_sources()
    assert load_entries("public")
    assert tagged_contents("public", "public_dataset_candidate")


def test_public_sources_are_license_gated(tmp_path):
    paths = Paths(str(tmp_path))
    manifest = validate_public_sources(paths)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["sources"] == []
    assert {source["license"] for source in data["sources"]}.issubset(ALLOWED_LICENSES)


def test_public_corpus_is_opt_in_without_local_files(tmp_path):
    paths = Paths(str(tmp_path))
    out = prepare_public_corpus(paths)
    assert out.read_text(encoding="utf-8") == ""


def test_fixture_dataset_validates(tmp_path):
    paths = Paths(str(tmp_path))
    fixture = prepare_fixtures(paths)
    assert validate_dataset(fixture) == fixture
    assert validate_dataset(paths.train_dataset) == paths.train_dataset
    assert validate_dataset(paths.val_dataset) == paths.val_dataset
    assert validate_dataset(paths.holdout_dataset) == paths.holdout_dataset


def test_prompt_text_uses_training_continuation_boundary():
    text = prompt_text([{"role": "user", "content": "hello"}])
    assert text.startswith("<bos>")
    assert text.endswith("<assistant_action>")
    assert "<dialogue>" in text
    assert "</dialogue>" not in text


def test_prompt_boundary_matches_message_text_before_assistant_target():
    assistant = "<action><tool>agent.finish</tool><content>hi</content></action>"
    messages = [{"role": "user", "content": "hello"}]
    trained = prompt_text(messages) + "\n" + assistant
    full = message_text(messages + [{"role": "assistant", "content": assistant}])
    assert full.startswith(trained + "\n</dialogue>")


def test_agent_corpus_default_has_required_mix():
    rows = generate_corpus(60000)
    assert len(rows) == 60000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert "direct_answer" in tags
    assert "docs_grounding" in tags
    assert "runtime_schema" in tags or "tool_trajectory" in tags
    assert tags.count("agentic") >= 10000
    assert tags.count("safety") >= 5000
    assert tags.count("preference") >= 3000
    assert all(row["meta"]["author_model"] == "none" for row in rows)


def test_normalize_action_extracts_first_xml_action():
    xml = "<action><tool>agent.finish</tool><content>ok</content></action>"
    text = f"noise {xml} trailing"
    assert parse_assistant_xml(normalize_action(text))["content"] == "ok"
    assert first_xml_action(text) == xml


def test_normalize_action_returns_raw_invalid_text():
    assert normalize_action("not json <eos>").strip() == "not json"


def test_behavioral_schema_rejects_invalid_action_shape():
    assert action_schema_error({}) == "action missing tool"
    assert action_schema_error({"tool": "fs.list", "path": "."}) == ""


def test_behavioral_buckets_report_pass_rates():
    cases = [
        {"bucket": "direct_answer", "passed": True, "valid_xml": True},
        {"bucket": "direct_answer", "passed": False, "valid_xml": True},
    ]
    assert bucket({"tags": ["kjxlkj", "confirmation"]}) == "kjxlkj_mutation_confirmation"
    assert bucket_rates(cases)["direct_answer"]["pass_rate"] == 0.5


def test_supervised_labels_mask_non_assistant_tokens():
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "<action><tool>agent.finish</tool><content>hi</content></action>"},
        ]
    }
    ids, labels = supervised_token_ids(DummyTokenizer(), row)
    assert len(ids) == len(labels)
    assert any(label == -100 for label in labels)
    assert any(label != -100 for label in labels)


def test_prepare_preferences_writes_pairs(tmp_path):
    paths = Paths(str(tmp_path))
    prepare_fixtures(paths)
    out = prepare_preferences(paths)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines
    assert {"messages", "chosen", "rejected", "source"}.issubset(lines[0])


class DummyTokenizer:
    def encode(self, text):
        return Encoded([ord(char) for char in text])


class Encoded:
    def __init__(self, ids):
        self.ids = ids
