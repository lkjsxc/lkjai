# Agent Schema

## XML Action

The model must return exactly one `<action>` block.

- Tags have no attributes.
- The `<tool>` child is required.
- `<reasoning>` is private trace material.
- `agent.finish` is the only normal successful terminator.

```xml
<action>
<reasoning>Need to inspect the workspace before answering.</reasoning>
<tool>fs.list</tool>
<path>.</path>
</action>
```

```xml
<action>
<reasoning>Need a visible planning step before tools.</reasoning>
<tool>agent.think</tool>
<content>Read docs, inspect code, then verify.</content>
</action>
```

```xml
<action>
<reasoning>The requested work is complete.</reasoning>
<tool>agent.finish</tool>
<content>Done.</content>
</action>
```

```xml
<action>
<reasoning>A kjxlkj mutation needs confirmation.</reasoning>
<tool>agent.request_confirmation</tool>
<summary>Update release notes?</summary>
<operation>resource.update_resource</operation>
<pending_tool>resource.update_resource</pending_tool>
<ref>release-notes</ref>
<body># Updated</body>
<is_private>false</is_private>
</action>
```

## Event Kinds

- `user`
- `assistant`
- `plan`
- `tool_call`
- `tool_result`
- `observation`
- `memory_write`
- `finish`
- `confirmation_request`
- `error`
