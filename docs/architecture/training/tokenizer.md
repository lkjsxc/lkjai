# Tokenizer Contract

## Goal

Keep prompt boundaries and XML-like action structure easy for the small model to
learn by making canonical tags atomic tokenizer tokens.

## Contract

- The tokenizer is a local byte-level BPE tokenizer.
- Base control tokens are special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`,
  and `<assistant_action>`.
- Canonical XML-like tags are added tokens, not special tokens.
- Every canonical XML-like tag must encode to exactly one token.
- Decoding with `skip_special_tokens=True` must keep XML-like tags.
- The final tokenizer vocabulary must not exceed the configured
  `TRAIN_VOCAB_SIZE`.

## Canonical Tags

Canonical tags cover model-facing prompt structure, agent action XML, runtime
context, and recurring first-party corpus sections.

- Dialogue: `<dialogue>`, `</dialogue>`, `<message>`, `</message>`,
  `<role>`, `</role>`, `<tool_name>`, `</tool_name>`, `<content>`,
  `</content>`.
- Runtime context: `<run>`, `</run>`, `<run_id>`, `</run_id>`, `<step>`,
  `</step>`, `<summary>`, `</summary>`, `<memories>`, `</memories>`,
  `<events>`, `</events>`, `<event>`, `</event>`, `<kind>`, `</kind>`.
- Task prompts: `<task>`, `</task>`, `<request>`, `</request>`,
  `<context>`, `</context>`, `<constraints>`, `</constraints>`.
- Assistant actions: `<action>`, `</action>`, `<reasoning>`, `</reasoning>`,
  `<tool>`, `</tool>`.
- Tool fields and corpus metadata sections: `<path>`, `</path>`, `<query>`,
  `</query>`, `<summary>`, `</summary>`, `<operation>`, `</operation>`,
  `<pending_tool>`, `</pending_tool>`, `<ref>`, `</ref>`, `<body>`,
  `</body>`, `<is_private>`, `</is_private>`, `<case>`, `</case>`,
  `<schema>`, `</schema>`, `<scenario>`, `</scenario>`, `<skill>`,
  `</skill>`, `<source>`, `</source>`, `<title>`, `</title>`,
  `<snippet>`, `</snippet>`, `<angle>`, `</angle>`, `<audience>`,
  `</audience>`, `<policy>`, `</policy>`, `<first>`, `</first>`,
  `<error>`, `</error>`, `<blocker>`, `</blocker>`, `<draft>`, `</draft>`,
  `<session>`, `</session>`, `<mode>`, `</mode>`.

## Prompt Serialization

- Model-facing message serialization must use paired tags without attributes.
- User, system, assistant, and tool roles are represented as text inside
  `<role>...</role>`.
- Tool names are represented as text inside `<tool_name>...</tool_name>`.
- Prompt construction still ends with `<assistant_action>\n`.
- Assistant targets remain exactly one XML action.

## Verification

- Unit tests must assert every canonical XML-like tag encodes to one token.
- Unit tests must assert model-facing prompts contain no `<message role=...>`
  attribute tags.
- Full acceptance uses the Compose verify profile.
