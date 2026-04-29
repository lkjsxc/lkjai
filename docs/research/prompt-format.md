# Prompt Format Research

## Policy

- Use paired XML-like tags for prompt structure and assistant actions.
- Keep model-facing tags attribute-free so each canonical tag can be one
  tokenizer token.
- Assistant action tags must not use attributes.
- JSON should be rare in training data and not used for normal actions.

## Why

- Paired tags are easier for small models to segment than long free-form prose.
- Atomic tags reduce the number of decisions needed to open or close an XML-like
  section.
- Child-tag actions avoid quoting and brace-balancing failures.
- Runtime validation still supplies typed fields before tools execute.

## References

- Anthropic prompt engineering recommends XML tags for prompt structure:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags>
- OpenAI structured output guidance recommends strict typed JSON output when the
  application needs validated fields:
  <https://developers.openai.com/api/docs/guides/structured-outputs>
