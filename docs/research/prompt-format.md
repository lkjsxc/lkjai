# Prompt Format Research

## Current Policy

- Use paired XML-like tags to segment complex runtime prompt context.
- Use strict JSON for assistant actions.
- Do not ask the model to emit XML tool calls in the default runtime.

## Rationale

- Tagged prompt sections make run metadata, summaries, memories, and events
  easier to separate in long prompts.
- JSON actions are easier for the Rust runtime to validate before tool
  execution.
- Schema-like JSON output matches production structured-output practice better
  than free-form text.

## References

- Anthropic prompt engineering docs recommend XML tags for clear prompt
  structure: <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags>
- OpenAI structured output guidance recommends strict schema-constrained JSON
  when applications need typed outputs:
  <https://developers.openai.com/api/docs/guides/structured-outputs>
