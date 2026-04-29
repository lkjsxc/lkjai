BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<assistant_action>",
]

XML_TAG_NAMES = [
    "dialogue",
    "message",
    "role",
    "tool_name",
    "content",
    "run",
    "run_id",
    "step",
    "summary",
    "memories",
    "events",
    "event",
    "kind",
    "task",
    "request",
    "context",
    "constraints",
    "action",
    "reasoning",
    "tool",
    "path",
    "query",
    "command",
    "url",
    "operation",
    "pending_tool",
    "ref",
    "id",
    "body",
    "is_private",
    "is_favorite",
    "current_resource_id",
    "alias",
    "case",
    "schema",
    "scenario",
    "skill",
    "source",
    "title",
    "snippet",
    "angle",
    "audience",
    "policy",
    "first",
    "error",
    "blocker",
    "draft",
    "session",
    "mode",
    "args",
]

XML_TAG_TOKENS = [
    token
    for name in XML_TAG_NAMES
    for token in (f"<{name}>", f"</{name}>")
]


def bpe_vocab_size(final_vocab_size: int) -> int:
    reserved = len(XML_TAG_TOKENS)
    return max(len(BASE_SPECIAL_TOKENS), final_vocab_size - reserved)
