# Agent Action SFT Template

Owner: `tools/kimi-corpus/prompts/agent-action-sft.md`.
State: prompt asset.

Generate one JSONL row with `messages`, `tags`, and `meta`.
The assistant target must contain exactly one `<action>` block and one allowed
tool. Mutating resource operations must request confirmation instead of
executing directly.
