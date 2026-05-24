# Agent Action SFT Template

Owner: `tools/kimi-corpus/prompts/agent-action-sft.md`.
State: prompt asset.

Generate English-only JSONL rows with `messages`, `tags`, and `meta`.
The assistant target must contain exactly one `<action>` block and one allowed
tool. Mutating resource operations must request confirmation instead of
executing directly.

Honor the request constraints exactly:

- Return only JSON, never markdown.
- Do not use Kimi Code CLI tools; synthesize only the requested corpus rows.
- If multi-turn is required, include at least two user turns, at least two
  assistant XML actions, and at least one tool message with fixture-derived
  output.
- If compaction is required, start the row with an English system message whose
  content begins `Conversation summary: `, set `meta.compaction` to true, and
  include the `compacted-context` tag.
- Use the `lkjai-agent-jsonl` schema and keep all content commercial-safe and
  workspace-safe.
