import json

from lkjai_train.behavioral import bucket
from lkjai_train.corpus import generate_corpus
from lkjai_train.corpus_source import load_entries


def test_agentic_source_files_are_tagged_json_arrays():
    assert load_entries("agentic_plan")
    assert load_entries("agentic_tools")
    assert load_entries("agentic_revision")
    assert load_entries("docs_grounding")


def test_agentic_rows_have_plan_event():
    from lkjai_train.corpus_agentic import plan_rows

    rows = plan_rows(100)
    assert any('"kind":"plan"' in json.dumps(row.get("messages", [])) for row in rows)


def test_agentic_rows_have_tool_observation_sequence():
    from lkjai_train.corpus_agentic import chain_rows

    rows = chain_rows(50)
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert "tool" in roles
        assert "assistant" in roles


def test_corpus_mix_counts_for_30k():
    rows = generate_corpus(6000)
    assert len(rows) == 6000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert tags.count("docs_grounding") >= 3500
    assert tags.count("runtime_schema") >= 900
    assert tags.count("tool_trajectory") >= 500
    assert tags.count("agentic") == 0
    assert tags.count("general_reasoning") == 0


def test_unique_scenario_ids_for_30k():
    rows = generate_corpus(6000)
    ids = [row["meta"]["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_default_corpus_has_no_llm_authored_rows():
    rows = generate_corpus(6000)
    for row in rows:
        assert row["meta"]["author_model"] == "none"
        assert row["meta"]["author_type"] != "llm-curated"


def test_behavioral_bucket_includes_agentic():
    assert bucket({"tags": ["agentic", "planning"]}) == "agentic_planning"
    assert bucket({"tags": ["agentic", "tool_chain"]}) == "agentic_tool_chain"
    assert bucket({"tags": ["agentic", "revision"]}) == "agentic_revision"
    assert bucket({"tags": ["agentic"]}) == "agentic_multi_turn"
