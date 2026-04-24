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
    assert any(
        m.get("role") == "assistant" and json.loads(m.get("content", "{}")).get("kind") == "plan"
        for row in rows
        for m in row.get("messages", [])
    )


def test_agentic_rows_have_tool_observation_sequence():
    from lkjai_train.corpus_agentic import chain_rows

    rows = chain_rows(50)
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert "tool" in roles
        assert "assistant" in roles


def test_active_agentic_rows_have_multi_turn_structure():
    from lkjai_train.corpus_agentic_active import agentic_active_rows

    rows = agentic_active_rows(100)
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert "tool" in roles
        assert "assistant" in roles
        assert "user" in roles
        assert row["meta"]["author_model"] == "none"
        assert row["meta"]["provenance"] == "repo-derived"


def test_corpus_mix_counts_for_60k():
    rows = generate_corpus(60000)
    assert len(rows) == 60000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert tags.count("docs_grounding") >= 14000
    assert tags.count("agentic") >= 10000
    assert tags.count("runtime_schema") >= 8000
    assert tags.count("tool_trajectory") >= 5000
    assert tags.count("safety") >= 5000
    assert tags.count("preference") >= 3000


def test_unique_scenario_ids_for_60k():
    rows = generate_corpus(60000)
    ids = [row["meta"]["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_default_corpus_has_no_llm_authored_rows():
    rows = generate_corpus(60000)
    for row in rows:
        assert row["meta"]["author_model"] == "none"
        assert row["meta"]["author_type"] != "llm-curated"


def test_behavioral_bucket_includes_agentic():
    assert bucket({"tags": ["agentic", "planning"]}) == "agentic_planning"
    assert bucket({"tags": ["agentic", "tool_chain"]}) == "agentic_tool_chain"
    assert bucket({"tags": ["agentic", "revision"]}) == "agentic_revision"
    assert bucket({"tags": ["agentic"]}) == "agentic_multi_turn"
