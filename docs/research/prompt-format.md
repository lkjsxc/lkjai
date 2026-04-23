# Prompt Format Research

## Policy

- Use paired XML-like tags for model-facing prompt structure.
- Use strict JSON for assistant actions.
- Do not use JSON blobs as the main prompt framing format.

## Why

- Paired tags are easier for small models to segment than long free-form prose.
- Strict JSON outputs remain the best runtime contract for typed tool actions.
- The right split is XML-like input structure plus JSON action output.

## References

- Anthropic prompt engineering recommends XML tags for prompt structure:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags>
- OpenAI structured output guidance recommends strict typed JSON output when the
  application needs validated fields:
  <https://developers.openai.com/api/docs/guides/structured-outputs>
