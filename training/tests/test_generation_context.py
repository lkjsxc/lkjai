from lkjai_train.generation import agent_context_messages, latest_user_event, normalize_messages


def test_raw_user_prompt_stays_raw():
    messages = normalize_messages([{"role": "user", "content": "What is 2+3?"}])
    assert messages == [{"role": "user", "content": "What is 2+3?"}]


def test_latest_user_event_extracts_tagged_context():
    content = "<events><event><kind>user</kind><content>What is lkjai?</content></event></events>"
    assert latest_user_event(content) == "What is lkjai?"


def test_agent_context_messages_include_tool_observation():
    content = "<events><event><kind>user</kind><content>Search resources.</content></event><event><kind>observation</kind><content>release-notes</content></event></events>"
    messages = agent_context_messages(content)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == "release-notes"


def test_agent_context_messages_preserve_empty_observation():
    content = "<events><event><kind>user</kind><content>Search resources.</content></event><event><kind>observation</kind><content></content></event></events>"
    messages = agent_context_messages(content)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"] == ""
