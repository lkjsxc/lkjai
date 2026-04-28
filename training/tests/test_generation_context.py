import torch

from lkjai_train.generation import agent_context_messages, choose_token, latest_user_event, normalize_messages


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


def test_sampling_falls_back_when_logits_are_not_finite():
    logits = torch.tensor([[float("nan"), 2.0, float("inf")]])
    assert choose_token(logits, 0.2).shape == (1, 1)


def test_sampling_handles_half_precision_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    token = choose_token(logits, 0.2)
    assert int(token.item()) in {0, 1, 2}
