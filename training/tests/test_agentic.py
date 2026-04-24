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
    rows = generate_corpus(30000)
    assert len(rows) == 30000
    tags = [tag for row in rows for tag in row.get("tags", [])]
    assert tags.count("agentic") >= 9000
    assert tags.count("direct_answer") + tags.count("general_reasoning") >= 6000
    assert tags.count("safety") >= 3000
    assert tags.count("kjxlkj") >= 3500
    assert tags.count("tool_trajectory") + tags.count("workspace_tool") + tags.count("runtime_tool") >= 2500
    assert tags.count("docs_grounding") >= 2000


def test_unique_scenario_ids_for_30k():
    rows = generate_corpus(30000)
    ids = [row["meta"]["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_behavioral_bucket_includes_agentic():
    assert bucket({"tags": ["agentic", "planning"]}) == "agentic_planning"
    assert bucket({"tags": ["agentic", "tool_chain"]}) == "agentic_tool_chain"
    assert bucket({"tags": ["agentic", "revision"]}) == "agentic_revision"
    assert bucket({"tags": ["agentic"]}) == "agentic_multi_turn"
