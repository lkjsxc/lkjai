import json
from pathlib import Path

import pytest

from lkjai_train.cli import train_settings
from lkjai_train.corpus_source import tagged_contents
from lkjai_train.model_presets import MODEL_PRESETS
from lkjai_train.paths import Paths
import lkjai_train.public_import as public_import


def test_scratch_40m_settings_are_available(monkeypatch):
    monkeypatch.delenv("TRAIN_CONFIG", raising=False)
    monkeypatch.delenv("TRAIN_MODEL_PRESET", raising=False)
    settings = train_settings("scratch-40m")
    assert settings.model_preset == "scratch-40m"
    assert settings.model_name == "lkjai-scratch-40m"
    assert settings.sequence_len == 1024
    assert settings.layers == 10
    assert settings.hidden_size == 576
    assert settings.heads == 8
    assert settings.kv_heads == 2
    assert settings.ffn_size == 1536
    assert MODEL_PRESETS["scratch-40m"][7] == 400000
    assert approx_parameter_count(settings) == 39_567_168


def test_json_training_config_drives_agent_defaults(monkeypatch):
    config = Path(__file__).resolve().parents[2] / "configs" / "training" / "scratch_40m_12h.json"
    monkeypatch.setenv("TRAIN_CONFIG", str(config))
    monkeypatch.delenv("TRAIN_MAX_OPTIMIZER_STEPS", raising=False)
    monkeypatch.delenv("TRAIN_MODEL_PRESET", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    settings = train_settings("agent")

    assert settings.model_name == "lkjai-scratch-40m"
    assert settings.max_optimizer_steps == 400000
    assert settings.validate_every_optimizer_steps == 3000
    assert settings.save_latest_every_optimizer_steps == 3000
    assert settings.intermediate_save_every_optimizer_steps == 120000
    assert settings.keep_last_checkpoints == 8


def test_env_overrides_json_training_config(monkeypatch):
    config = Path(__file__).resolve().parents[2] / "configs" / "training" / "scratch_40m_12h.json"
    monkeypatch.setenv("TRAIN_CONFIG", str(config))
    monkeypatch.setenv("TRAIN_MAX_OPTIMIZER_STEPS", "7")
    monkeypatch.setenv("TRAIN_BATCH_POLICY", "fixed")

    settings = train_settings("agent")

    assert settings.max_optimizer_steps == 7
    assert settings.batch_policy == "fixed"


def test_public_candidates_are_not_active_sources_by_default():
    active = tagged_contents("public", "public_dataset")
    candidates = tagged_contents("public", "public_dataset_candidate")
    review_only = tagged_contents("public", "public_dataset_review_only")
    assert active == []
    assert {item["name"] for item in candidates} == {
        "oasst1-en",
        "oasst2-en",
        "smol-smoltalk",
        "hermes-function-calling-v1",
    }
    assert {item["name"] for item in review_only} == {
        "databricks-dolly-15k",
        "xlam-function-calling-60k",
    }


def test_public_pretrain_has_active_permissive_english_source():
    active = tagged_contents("public-pretrain", "public_pretrain_dataset")
    references = tagged_contents("public-pretrain", "public_pretrain_reference_only")
    assert {item["name"] for item in active} == {"cosmopedia-en"}
    assert active[0]["license"] == "Apache-2.0"
    assert active[0]["language"] == "en"
    assert {item["name"] for item in references} == {"fineweb-reference", "dolma-reference"}


def test_public_active_sources_must_use_pinned_permissive_license(tmp_path, monkeypatch):
    paths = Paths(str(tmp_path))
    monkeypatch.setattr(public_import, "public_sources", lambda: [public_source("CC-BY-4.0", "abc123")])
    with pytest.raises(ValueError, match="disallowed license"):
        public_import.validate_public_sources(paths)

    monkeypatch.setattr(public_import, "public_sources", lambda: [public_source("Apache-2.0", "main")])
    with pytest.raises(ValueError, match="pin an immutable revision"):
        public_import.validate_public_sources(paths)


def test_tracked_agent_prompt_preserves_tool_action_schema():
    prompt_path = Path(__file__).resolve().parents[2] / "apps" / "runtime" / "prompts" / "codex-40m-system.txt"
    text = prompt_path.read_text(encoding="utf-8")
    assert "<tool>agent.finish</tool>" in text
    assert "<tool>agent.request_confirmation</tool>" in text
    assert "Do not repeat the same failed non-terminal action" in text
    assert "<type>finish</type>" not in text


def test_public_source_manifest_is_empty_when_only_candidates_exist(tmp_path):
    paths = Paths(str(tmp_path))
    manifest = public_import.validate_public_sources(paths)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["sources"] == []


def public_source(license_name: str, revision: str) -> dict:
    return {
        "name": "candidate",
        "license": license_name,
        "source_url": "https://example.invalid/dataset",
        "revision": revision,
        "local_file": "candidate.jsonl",
    }


def approx_parameter_count(settings) -> int:
    head_dim = settings.hidden_size // settings.heads
    kv_dim = settings.kv_heads * head_dim
    attention = (settings.hidden_size * settings.hidden_size * 2) + (settings.hidden_size * kv_dim * 2)
    ffn = 3 * settings.hidden_size * settings.ffn_size
    norms = 2 * settings.hidden_size
    per_layer = attention + ffn + norms
    return (settings.vocab_size * settings.hidden_size) + (settings.layers * per_layer) + settings.hidden_size
